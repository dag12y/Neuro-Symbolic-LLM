"""parser/student/train.py

Fine-tunes the student parser on (input, json_label) pairs.

HOW TO RUN:
    python parser/student/train.py --train-file data/student_train.jsonl
    python parser/student/train.py --eval-only --model-dir models/distil_parser
"""

from __future__ import annotations

import argparse
import io
import json
import math
import platform
import re
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from parser.semantic.dataset import (
    StudentBatchCollator,
    StudentDataset,
    split_pairs,
    validate_student_pair,
    verify_split_overlap,
)
from parser.student.student_prompt import STUDENT_PROMPT
from versioning import Registry, canonical_json, export_manifest, hash_file
from versioning.checkpoint import checkpoint_save, register_training_provenance
from versioning.semantic import register_semantic_components

if TYPE_CHECKING:
    from parser.ontology.catalog import OntologyCatalog

# ── Load config ───────────────────────────────────────────────────────────────


def load_config(config_path: str = "configs/parser_config/student_config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


# ── Load training data ────────────────────────────────────────────────────────


class _HashedInput(io.FileIO):
    """Hash the exact bytes consumed by the UTF-8 JSONL reader in one pass."""

    def __init__(self, path: Path):
        super().__init__(path, "r")
        self.digest = sha256()

    def readinto(self, buffer):
        count = super().readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def load_pairs(
    train_file: str,
    *,
    catalog: OntologyCatalog | None = None,
    digests: dict[str, str] | None = None,
) -> list[dict]:
    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    raw = _HashedInput(train_path) if digests is not None else None
    reader = (
        io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8")
        if raw is not None
        else open(train_path, encoding="utf-8")
    )
    pairs = []
    with reader as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    pair = json.loads(line)
                    validate_student_pair(pair, catalog=catalog)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid pair at {train_path}:{line_number}: {error}"
                    ) from error
                pairs.append(pair)
    if raw is not None and digests is not None:
        digests[train_file] = raw.digest.hexdigest()
    print(f"  Loaded {len(pairs)} pairs from {train_file}")
    return pairs


def _catalog_training_prompt_version(catalog):
    """Return the active combined student prompt version."""
    from parser.student.student_prompt import build_student_prompt_spec

    return build_student_prompt_spec(
        "Checkpoint compatibility.",
        catalog=catalog,
    ).version


# ── LoRA setup ────────────────────────────────────────────────────────────────


def apply_lora(model, lora_config: dict):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as err:
        raise ImportError("Install peft: pip install peft") from err

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# Default weight filenames and shards used by the student loader.
_STUDENT_WEIGHT_FILE = re.compile(
    r"(?:model(?:-\d{5}-of-\d{5})?\.safetensors|"
    r"pytorch_model(?:-\d{5}-of-\d{5})?\.bin|"
    r"model\.safetensors\.index\.json|pytorch_model\.bin\.index\.json)"
)


# ── Save model ─────────────────────────────────────────────────────────────────


def save_model(
    model,
    tokenizer,
    output_dir: str,
    metadata: dict,
    training_manifest: bytes | None = None,
) -> None:
    """Save model, tokenizer, and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with checkpoint_save(output_path, training_manifest):
        # The student loader requires this file. Retire the previous metadata
        # before changing weights so an interrupted overwrite cannot reuse it.
        metadata_path = output_path / "training_metadata.json"
        metadata_path.unlink(missing_ok=True)
        # Transformers can prefer a previous unsharded file to a new shard index.
        for path in output_path.iterdir():
            if path.is_file() and _STUDENT_WEIGHT_FILE.fullmatch(path.name):
                path.unlink()
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        with metadata_path.open("w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2)

    print(f"\n  Model saved to: {output_dir}")
    print("  Files saved:")
    for file_path in sorted(output_path.iterdir()):
        size = file_path.stat().st_size / 1024
        if size > 1024:
            print(f"    - {file_path.name} ({size / 1024:.1f} MB)")
        else:
            print(f"    - {file_path.name} ({size:.1f} KB)")


def _training_manifest(
    catalog: OntologyCatalog,
    settings: dict,
    train_sha256: str | None,
    val_sha256: str | None,
    train_data: StudentDataset,
    val_data: StudentDataset,
) -> bytes:
    """Capture only digests of inputs, effective token sequences and source code."""
    root = Path(__file__).resolve().parents[2]
    packages = {}
    for package in ("torch", "transformers", "peft", "tokenizers", "PyYAML"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    datasets = {}
    for name, dataset in (("training", train_data), ("validation", val_data)):
        digest = None
        if hasattr(dataset, "examples"):
            digest = sha256()
            for example in dataset.examples:
                digest.update(canonical_json(example) + b"\n")
        datasets[name] = {
            "count": len(dataset),
            "ordered_token_examples_sha256": digest.hexdigest() if digest else None,
        }
    with TemporaryDirectory() as directory:
        registry = Registry(directory)
        ids = register_semantic_components(
            registry, project_root=root, effective_catalog=catalog
        )
        ontology_id = ids.effective_catalog or ids.ontology_bundle
        assert ontology_id is not None
        training_id = register_training_provenance(
            registry,
            student_prompt_id=ids.prompts["student"],
            ontology_id=ontology_id,
            details={
                "train_file_sha256": train_sha256,
                "validation_file_sha256": val_sha256,
                "datasets": datasets,
                "settings": settings,
                "code_sha256": {
                    path: hash_file(root / path)
                    for path in (
                        "parser/student/train.py",
                        "parser/semantic/dataset.py",
                        "parser/semantic/catalog_student.py",
                        "parser/ontology/catalog.py",
                    )
                },
                "python_version": platform.python_version(),
                "packages": packages,
                "cuda_build": torch.version.cuda,
                "unresolved": [
                    "base weights and tokenizer byte identities",
                    "GPU kernels, device placement, sampler order and RNG states",
                    "optimizer state and full environment",
                ],
            },
        )
        return export_manifest(registry, training_id)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    model_dir: str,
    test_sentences: list[str] | None = None,
    max_tokens: int = 3072,
) -> None:
    """Generate and validate predictions for explicitly supplied sources."""
    from parser.semantic.semantic_parser import DistilledSemanticParser

    if not test_sentences:
        raise ValueError(
            "Supply explicit evaluation sources; no synthetic default benchmark"
        )
    parser = DistilledSemanticParser.from_pretrained(
        model_dir, max_new_tokens=max_tokens
    )
    valid = 0
    for sentence in test_sentences:
        try:
            result = parser.parse(sentence)
            print(result.model_dump_json())
            valid += 1
        except (ValueError, RuntimeError) as error:
            print(f"Rejected prediction: {error}")
    print(f"Validated documents: {valid}/{len(test_sentences)}")
    print("No expected labels supplied; semantic accuracy was not measured.")


# ── Main training ─────────────────────────────────────────────────────────────


def train(
    train_file: str = "data/student_train.jsonl",
    config_path: str = "configs/parser_config/student_config.yaml",
    output_dir: str | None = None,
    method: str = "full",
    val_file: str | None = None,
) -> str:
    config = load_config(config_path)
    from parser.ontology.catalog import load_manifest_catalog
    from parser.semantic.schema import SEMANTIC_CONTRACT_VERSION

    contract_version = config["data"].get("contract_version")
    if contract_version != SEMANTIC_CONTRACT_VERSION:
        raise ValueError(
            f"Student training requires contract version {SEMANTIC_CONTRACT_VERSION}"
        )
    manifest = config["data"].get("ontology_bundle_manifest")
    if not isinstance(manifest, str) or not manifest.strip():
        raise ValueError("Document training requires ontology_bundle_manifest")
    catalog, _ = load_manifest_catalog(Path(manifest))
    training_prompt_version = _catalog_training_prompt_version(catalog)

    base_model = config["student"]["base_model"]
    val_split = config["data"]["val_split"]
    max_length = config["data"]["max_length"]
    if max_length == "auto":
        max_length = None
    train_cfg = config["training"]
    lora_cfg = config.get("lora", {})

    accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
    if (
        isinstance(accumulation_steps, bool)
        or not isinstance(accumulation_steps, int)
        or accumulation_steps < 1
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    bf16 = train_cfg.get("bf16", False)
    if not isinstance(bf16, bool):
        raise ValueError("bf16 must be a boolean")
    use_bf16 = bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if bf16 and not use_bf16:
        print("  BF16 is unavailable on the current device; using float32")

    if output_dir is None:
        output_dir = config["student"]["output_dir"]

    torch.manual_seed(train_cfg.get("seed", 42))

    print("\n" + "=" * 70)
    print("STUDENT PARSER TRAINING")
    print("=" * 70)
    print(f"  Method:          {method}")
    print(f"  Base model:      {base_model}")
    print(f"  Training data:   {train_file}")
    print(f"  Output dir:      {output_dir}")

    # Step 1: Load training pairs
    print("\nSTEP 1: Loading training data")
    input_digests: dict[str, str] = {}
    all_pairs = load_pairs(train_file, catalog=catalog, digests=input_digests)

    if len(all_pairs) == 0:
        raise ValueError(f"No valid pairs found in {train_file}")

    if val_file is not None:
        train_pairs = all_pairs
        val_pairs = load_pairs(val_file, catalog=catalog, digests=input_digests)
    else:
        train_pairs, val_pairs, _ = split_pairs(
            all_pairs,
            train_ratio=1 - val_split,
            val_ratio=val_split,
            test_ratio=0,
            seed=train_cfg.get("seed", 42),
        )
    verify_split_overlap(train_pairs, val_pairs, [])
    print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Step 2: Load tokenizer
    print("\nSTEP 2: Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Step 3: Build datasets
    print("\nSTEP 3: Building datasets")
    train_dataset = StudentDataset(train_pairs, tokenizer, max_length, catalog=catalog)
    val_dataset = StudentDataset(val_pairs, tokenizer, max_length, catalog=catalog)
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    required_length = max(
        getattr(dataset, "required_max_length", 0)
        for dataset in (train_dataset, val_dataset)
    )
    context_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(context_limit, int) and required_length > context_limit:
        raise ValueError(
            f"Document pairs need {required_length} tokens; tokenizer context limit is {context_limit}"
        )
    print(f"  Longest complete prompt and label: {required_length} tokens")

    if not train_dataset:
        raise ValueError(
            "No training examples fit max_length; increase it or check the data"
        )
    if val_pairs and not val_dataset:
        raise ValueError(
            "No validation examples fit max_length; increase it or check the data"
        )

    collator = StudentBatchCollator(tokenizer.pad_token_id)

    # Step 4: Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    # Step 5: Load model
    print("\nSTEP 4: Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )
    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Capture resolved settings before PEFT wraps the base model.
    revision = getattr(model.config, "_commit_hash", None)
    resolved_revision = (
        revision
        if isinstance(revision, str)
        and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision)
        else None
    )
    public_model_id = (
        base_model
        if resolved_revision
        and isinstance(base_model, str)
        and re.fullmatch(r"[A-Za-z0-9_-][\w.-]*/[A-Za-z0-9_-][\w.-]*", base_model)
        and not base_model.startswith(("hf_", "sk-", "ghp_"))
        and not Path(base_model).exists()
        else None
    )
    config_bytes = yaml.safe_dump(config, sort_keys=True).encode()
    settings = {
        "method": "lora" if method == "lora" else "full",
        "base_model": public_model_id,
        "requested_model_identifier_sha256": sha256(str(base_model).encode()).hexdigest(),
        "student_prompt_version": training_prompt_version,
        "device_type": model.device.type,
        "device_map": "auto",
        "gradient_checkpointing": True,
        "base_model_revision": resolved_revision,
        "config_sha256": sha256(config_bytes).hexdigest(),
        "config_scope": "parsed YAML reserialized by PyYAML, sorted keys",
        "epochs": train_cfg["epochs"],
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
        "gradient_accumulation_steps": accumulation_steps,
        "optimizer": "AdamW",
        "scheduler": train_cfg.get("lr_scheduler", "constant"),
        "warmup_steps": 0,
        "seed": int(train_cfg.get("seed", 42)),
        "dtype": "bfloat16" if use_bf16 else "float32",
        "bf16_requested": bf16,
        "max_length": max_length,
        "split": "explicit" if val_file is not None else "seeded",
        "validation_fraction": None if val_file is not None else val_split,
    }
    if method == "lora":
        targets = lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        common_targets = {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        }
        lora_numbers = {
            key: value if type(value) in (int, float) else None
            for key, value in {
                "r": lora_cfg.get("r", 16),
                "alpha": lora_cfg.get("alpha", 32),
                "dropout": lora_cfg.get("dropout", 0.05),
            }.items()
        }
        settings["lora"] = {
            **lora_numbers,
            "target_modules": (
                targets
                if isinstance(targets, list)
                and all(isinstance(t, str) and t in common_targets for t in targets)
                else None
            ),
            "target_modules_sha256": sha256(canonical_json(targets)).hexdigest(),
        }
    # Step 6: Apply method
    if method == "lora":
        print("\nSTEP 5: Applying LoRA adapters")
        model = apply_lora(model, lora_cfg)
    else:
        print("\nSTEP 5: Full fine-tuning")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"  Trainable: {trainable / 1e9:.2f}B ({100 * trainable / model.num_parameters():.1f}%)"
        )

    # Step 7: Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    scheduler = get_scheduler(
        train_cfg.get("lr_scheduler", "constant"),
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=updates_per_epoch * train_cfg["epochs"],
    )
    settings["optimizer_defaults"] = {
        "betas": list(optimizer.defaults["betas"]),
        "eps": optimizer.defaults["eps"],
        "weight_decay": optimizer.defaults["weight_decay"],
        "amsgrad": optimizer.defaults["amsgrad"],
    }
    settings["updates_per_epoch"] = updates_per_epoch
    training_manifest = _training_manifest(
        catalog,
        settings,
        input_digests.get(train_file),
        input_digests.get(val_file) if val_file is not None else None,
        train_dataset,
        val_dataset,
    )
    optimizer.zero_grad()
    model.train()

    # Step 8: Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for epoch in range(train_cfg["epochs"]):
        print(f"Epoch {epoch + 1}/{train_cfg['epochs']}")

        accumulated_tokens = 0
        for batch_index, batch in enumerate(train_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            # Weight by supervised tokens so unequal sequence lengths and the
            # final partial window match the gradient of a combined batch.
            batch_tokens = (batch["labels"][:, 1:] != -100).sum().item()
            (loss * batch_tokens).backward()
            accumulated_tokens += batch_tokens
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(
                train_loader
            ):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_tokens)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulated_tokens = 0

            # Calculate accuracy
            logits = outputs.logits
            labels = batch["labels"]
            shifted_labels = labels[:, 1:]
            shifted_logits = logits[:, :-1, :]
            predictions = shifted_logits.argmax(dim=-1)

            mask = shifted_labels != -100
            correct = (predictions == shifted_labels) & mask
            total = mask.sum().item()

            total_loss += loss.item()
            if total > 0:
                total_correct += correct.sum().item()
                total_tokens += total

            global_step += 1

            if global_step % 50 == 0:
                avg_loss = total_loss / 50
                accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
                print(
                    f"  Batch {global_step}: loss = {avg_loss:.4f}, acc = {accuracy:.4f}"
                )
                total_loss = 0.0
                total_correct = 0
                total_tokens = 0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                labels = batch["labels"]
                shifted_labels = labels[:, 1:]
                shifted_logits = logits[:, :-1, :]
                predictions = shifted_logits.argmax(dim=-1)

                mask = shifted_labels != -100
                correct = (predictions == shifted_labels) & mask
                total = mask.sum().item()

                if total > 0:
                    val_correct += correct.sum().item()
                    val_tokens += total

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_correct / val_tokens if val_tokens > 0 else 0

        print(
            f"  Epoch {epoch + 1} complete: val_loss = {avg_val_loss:.4f}, val_acc = {val_acc:.4f}"
        )
        model.train()

    # Step 9: Save model
    print("\nSTEP 7: Saving model")

    if method == "lora":
        print("  Merging LoRA adapters...")
        model = model.merge_and_unload()

    metadata = {
        "base_model": base_model,
        "method": method,
        "train_file": train_file,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "config": config,
        "prompt_format": STUDENT_PROMPT,
        "target_contract_version": contract_version,
        "ontology_versions": list(catalog.versions),
        "prompt_version": training_prompt_version,
        "epochs": train_cfg["epochs"],
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
    }

    save_model(model, tokenizer, output_dir, metadata, training_manifest)

    return output_dir


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/student_train.jsonl")
    parser.add_argument("--val-file", help="Separate validation pairs JSONL file")
    parser.add_argument("--config", default="configs/parser_config/student_config.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--method", choices=["lora", "full"], default="full")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--test-sentence", action="append")
    parser.add_argument("--model-dir", default="models/semantic_document_parser")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.model_dir, args.test_sentence)
    else:
        train(
            train_file=args.train_file,
            val_file=args.val_file,
            config_path=args.config,
            output_dir=args.output_dir,
            method=args.method,
        )


if __name__ == "__main__":
    main()
