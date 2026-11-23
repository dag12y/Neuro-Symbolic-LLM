"""Concrete stage definitions and in-process producer adapters."""

from __future__ import annotations

import json
import re
import sys
from contextlib import redirect_stdout, suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from .execution import _file_identity
from .mining import validate_release
from .neural import build_a1_definition, validate_a1_definition
from .nodes import Node, reachable_nodes
from .symbolic import build_symbolic_definition, validate_symbolic_definition
from .workflow import PROJECT, composite, slots

EMBED_SOURCES = tuple(
    f"offline/embedding/{n}.py"
    for n in ("context_builder", "encoder", "template_embeddings", "run")
) + tuple(
    f"frozenllm/substrate/{n}.py"
    for n in (
        "substrate",
        "architecture",
        "loader",
        "interception",
        "torchax_backend",
        "torchax_models",
        "drift",
        "memory",
        "__init__",
    )
)
STUDENT_SOURCES = (
    "parser/student/train.py",
    "parser/student/student_prompt.py",
    "parser/semantic/dataset.py",
)
SETTINGS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "miner": {"min_support": int},
    "embedding": {
        "model": str,
        "max_length": int,
        "layer_idx": (int, type(None)),
        "symbolic_dim": int,
        "projection_seed": int,
    },
    "a1-train": {},
    "a1-eval": {"split": str},
    "student-train": {"method": str},
    "student-eval": {},
}


def validate_options(stage: str, values: dict[str, Any]) -> None:
    spec = SETTINGS[stage]
    if set(values) != set(spec) or any(
        type(values[k]) not in (t if isinstance(t, tuple) else (t,))
        for k, t in spec.items()
    ):
        raise ValueError("Invalid stage options")
    if any(
        values.get(k, 1) < 1 for k in ("min_support", "max_length", "symbolic_dim")
    ) or (stage == "student-train" and values["method"] not in ("full", "lora")):
        raise ValueError("Invalid stage option value")


def stage_definition(
    stage: str, config: Path | None = None, bundle: Path | None = None
) -> tuple[Node, ...]:
    sources: tuple[str, ...]
    if stage.startswith("a1-"):
        assert config is not None
        base = build_a1_definition(config)
        sources = ("stages/observations.py",)
    elif stage.startswith("student-"):
        base = build_symbolic_definition(PROJECT, ontology_bundle=bundle)
        sources = STUDENT_SOURCES
    else:
        base = ()
        sources = EMBED_SOURCES
    leaves = tuple(
        Node(kind="source", name=p, payload=_file_identity(PROJECT / p))
        for p in sources
    )
    edges = {n.name: n for n in leaves}
    if base:
        edges["base"] = base[-1]
    return (
        *base,
        *leaves,
        composite(f"workflow/{stage}/definition", "definition", edges),
    )


def validate_definition(stage: str, defs: tuple[Node, ...]) -> None:
    if stage == "miner":
        validate_release(defs)
        return
    root = defs[-1]
    index = {n.id: n for n in defs}
    edges = slots(root, index)
    expected = EMBED_SOURCES
    if stage.startswith(("a1-", "student-")):
        if "base" not in edges:
            raise ValueError("Missing base definition")
        base = edges["base"]
        closure = reachable_nodes((base.id,), index.__getitem__)
        if stage.startswith("a1-"):
            validate_a1_definition(closure, base.id)
            expected = ("stages/observations.py",)
        else:
            validate_symbolic_definition(closure, base.id)
            expected = STUDENT_SOURCES
    if (
        set(edges) != (set(expected) | ({"base"} if stage != "embedding" else set()))
        or any(edges[k].name != k or edges[k].kind != "source" for k in expected)
        or root != composite(f"workflow/{stage}/definition", "definition", edges)
    ):
        raise ValueError("Invalid stage definition")


def versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for package in (
        "numpy",
        "jax",
        "jaxlib",
        "torch",
        "torchax",
        "optax",
        "transformers",
        "datasets",
    ):
        with suppress(PackageNotFoundError):
            result[package] = version(package)
    return result


def substrate_observation(substrate: Any) -> dict[str, Any]:
    tokenizer = substrate.tokenizer
    cfg = getattr(getattr(substrate, "_model", None), "config", None)
    return {
        "model_revision_reported": getattr(cfg, "_commit_hash", None),
        "tokenizer_revision_reported": getattr(tokenizer, "init_kwargs", {}).get(
            "_commit_hash"
        ),
        "tokenizer_class": type(tokenizer).__name__,
        "layers": substrate.architecture.num_layers,
        "hidden_size": substrate.architecture.hidden_size,
    }


def embed(
    private: Path, out: Path, runtime: dict[str, Any], settings: dict[str, Any]
) -> None:
    from offline.embedding.run import generate_embeddings

    def loaded(encoder: Any) -> None:
        runtime.update(
            versions=versions(), substrate=substrate_observation(encoder.substrate)
        )

    with redirect_stdout(sys.stderr):
        generate_embeddings(
            private / "documents.jsonl",
            private / "patterns.jsonl",
            out / "embeddings.jsonl",
            **settings,
            observe_encoder=loaded,
        )


def a1(
    private: Path, out: Path, runtime: dict[str, Any], settings: dict[str, Any]
) -> None:
    from experiments.run_stage_a1 import main
    from stages.observations import observer

    from .neural_integrity import (
        inspect_a1_checkpoint,
        ordered_text_rows_node,
        token_blocks_node,
    )

    runtime.update(versions=versions(), datasets=[])

    def observe(event: str, **kw: Any) -> None:
        if event == "dataset":

            def value(node: Node) -> Any:
                return cast(dict[str, Any], node.to_dict()["payload"])["value"]

            runtime["datasets"].append(
                {
                    "domain": kw["domain"],
                    "split": kw["split"],
                    "rows": value(ordered_text_rows_node("rows", kw["rows"])),
                    "blocks": value(token_blocks_node("blocks", kw["blocks"])),
                }
            )
        elif event == "eval_prefix":
            selected = token_blocks_node("prefix", kw["blocks"])
            runtime["datasets"][-1]["evaluation_prefix"] = cast(
                dict[str, Any], selected.to_dict()["payload"]
            )["value"]
        elif event == "initial_parameters":
            import hashlib

            import numpy as np

            arrays = {}
            for layer, leaves in kw["params"].items():
                for name, weights in leaves.items():
                    array = np.asarray(weights)
                    arrays[f"{layer}/{name}"] = {
                        "shape": list(array.shape),
                        "dtype": array.dtype.str,
                        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                    }
            runtime["initial_parameters"] = arrays
        elif event == "substrate":
            runtime["substrate"] = substrate_observation(kw["substrate"])
        elif event == "evaluation":
            (out / "evaluation.json").write_text(
                json.dumps(kw["results"], allow_nan=False) + "\n"
            )

    token = observer.set(observe)
    try:
        args = ["--config", str(private / "config.yaml")]
        if "split" in settings:
            checkpoint = private / "checkpoint.npz"
            if inspect_a1_checkpoint(checkpoint).classification == "unknown":
                raise ValueError("Unrecognized evaluation checkpoint")
            args += ["--eval-checkpoint", str(checkpoint), "--split", settings["split"]]
        else:
            out.rmdir()  # RunWriter creates this private, empty directory itself.
            args += ["--output-dir", str(out.parent), "--run-name", out.name]
        with redirect_stdout(sys.stderr):
            if main(args):
                raise ValueError("A1 driver returned failure")
        for path in out.rglob("*.npz"):
            inspected = inspect_a1_checkpoint(path)
            expected = (
                "full_a1" if path.parent.name == "checkpoints" else "weights_only_a1"
            )
            if inspected.classification != expected:
                raise ValueError("Unexpected produced checkpoint layout")
    finally:
        observer.reset(token)


def student(
    private: Path, out: Path, runtime: dict[str, Any], settings: dict[str, Any]
) -> None:
    from parser.student.train import evaluate, train

    runtime["versions"] = versions()

    def loaded(model: Any, tokenizer: Any) -> None:
        runtime["model"] = {
            "class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "revision_reported": getattr(
                getattr(model, "config", None), "_commit_hash", None
            ),
            "tokenizer_revision_reported": getattr(tokenizer, "init_kwargs", {}).get(
                "_commit_hash"
            ),
        }

    if "method" in settings:
        val = private / "validation.jsonl"
        with redirect_stdout(sys.stderr):
            train(
                str(private / "train.jsonl"),
                str(private / "config.yaml"),
                str(out),
                settings["method"],
                str(val) if val.exists() else None,
                observe_model=loaded,
            )
    else:
        with (out / "evaluation.txt").open("x") as stream, redirect_stdout(stream):
            evaluate(
                str(private / "model"),
                (private / "sentences.txt").read_text(encoding="utf-8").splitlines(),
                observe_model=loaded,
            )


def validate_runtime(
    stage: str, runtime: dict[str, Any], defs: tuple[Node, ...], options: dict[str, Any]
) -> None:
    from .nodes import validate_id

    if stage == "miner":
        resolved = next(n for n in defs if n.name == "workflow/miner/resolution")
        value = cast(dict[str, Any], resolved.to_dict()["payload"])["value"]
        if (
            set(runtime)
            != {"commit", "package_tree_sha256", "steps", "engine_heads_reported"}
            or value["method"] != "local-git"
            or runtime["commit"] != value["commit"]
        ):
            raise ValueError("Miner execution identity differs from resolved source")
        heads = runtime["engine_heads_reported"]
        if (
            not isinstance(heads, dict)
            or set(heads) != {"petta", "hyperon_miner"}
            or not all(
                v is None
                or isinstance(v, str)
                and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", v)
                for v in heads.values()
            )
        ):
            raise ValueError("Invalid engine Git observations")
        validate_id(runtime["package_tree_sha256"])
        names = [
            "ingestion",
            "emission",
            "mining.admitted_support",
            "mining.exact_matches",
            "mining.build_pattern_records",
        ]
        steps = runtime["steps"]
        if (
            not isinstance(steps, list)
            or len(steps) != len(names)
            or any(
                not isinstance(s, dict)
                or set(s) != {"module", "code"}
                or s["module"] != n
                or type(s["code"]) is not int
                or s["code"] not in ((0, 2) if n == "ingestion" else (0,))
                for s, n in zip(steps, names, strict=True)
            )
        ):
            raise ValueError("Incomplete mining execution")
        return
    keys = {"versions"}
    if stage.startswith("student-"):
        keys.add("model")
        model = runtime.get("model")
        if (
            not isinstance(model, dict)
            or set(model)
            != {
                "class",
                "tokenizer_class",
                "revision_reported",
                "tokenizer_revision_reported",
            }
            or not all(isinstance(model[k], str) for k in ("class", "tokenizer_class"))
            or not all(
                isinstance(model[k], str | None)
                for k in ("revision_reported", "tokenizer_revision_reported")
            )
        ):
            raise ValueError("Invalid student model observations")
    if stage == "embedding" or stage.startswith("a1-"):
        keys.add("substrate")
    if stage.startswith("a1-"):
        keys.update(("datasets", "initial_parameters"))
    if (
        set(runtime) != keys
        or not isinstance(runtime["versions"], dict)
        or "python" not in runtime["versions"]
        or not all(isinstance(v, str) for v in runtime["versions"].values())
    ):
        raise ValueError("Missing runtime observations")
    if "substrate" in keys:
        obs = runtime["substrate"]
        if (
            not isinstance(obs, dict)
            or set(obs)
            != {
                "model_revision_reported",
                "tokenizer_revision_reported",
                "tokenizer_class",
                "layers",
                "hidden_size",
            }
            or not all(
                type(obs[k]) is int and obs[k] > 0 for k in ("layers", "hidden_size")
            )
            or not isinstance(obs["tokenizer_class"], str)
            or not all(
                isinstance(obs[k], str | None)
                for k in ("model_revision_reported", "tokenizer_revision_reported")
            )
        ):
            raise ValueError("Invalid loaded substrate observation")
    if "datasets" in keys:
        data = next(n for n in defs if n.name == "neural/a1/config/data")
        plan = cast(dict[str, Any], data.to_dict()["payload"])["value"]
        domains = plan["domains"]
        if stage == "a1-train":
            expected = [
                (d["name"], d[k])
                for d in domains
                for k in ("train_split", "eval_split")
            ]
            expected.extend((d["name"], d["test_split"]) for d in domains)
        else:
            split = options["split"]
            field = {
                "train": "train_split",
                "eval": "eval_split",
                "validation": "eval_split",
                "test": "test_split",
            }.get(split, "")
            expected = [(d["name"], d.get(field, split)) for d in domains]
        sets = runtime["datasets"]
        if not isinstance(sets, list) or len(sets) != len(expected):
            raise ValueError("Incomplete A1 dataset observations")
        params = runtime["initial_parameters"]
        if not isinstance(params, dict) or not params:
            raise ValueError("Missing adapter initialization")

        for name, arr in params.items():
            if (
                not re.fullmatch(r"layer_[0-9]+/[AB]", name)
                or name[:-1] + ("A" if name[-1] == "B" else "B") not in params
                or not isinstance(arr, dict)
                or set(arr) != {"shape", "dtype", "sha256"}
                or not isinstance(arr["dtype"], str)
                or not isinstance(arr["shape"], list)
                or len(arr["shape"]) != 2
                or not all(type(n) is int and n > 0 for n in arr["shape"])
            ):
                raise ValueError("Invalid adapter initialization")
            validate_id(arr["sha256"])
        for i, (d, pair) in enumerate(zip(sets, expected, strict=True)):
            evaluated = stage == "a1-eval" or i >= 2 * len(domains) or i % 2 == 1
            if (
                not isinstance(d, dict)
                or set(d)
                != (
                    {"domain", "split", "rows", "blocks"}
                    | ({"evaluation_prefix"} if evaluated else set())
                )
                or (d["domain"], d["split"]) != pair
            ):
                raise ValueError("Unexpected A1 dataset split")
            rows, blocks = d["rows"], d["blocks"]
            if (
                not isinstance(rows, dict)
                or set(rows) != {"row_count", "framed_sha256"}
                or type(rows["row_count"]) is not int
                or rows["row_count"] < 1
            ):
                raise ValueError("Invalid A1 text observation")
            validate_id(rows["framed_sha256"])
            if not isinstance(blocks, dict) or set(blocks) != {
                "shape",
                "dtype",
                "byte_order",
                "layout",
                "byte_length",
                "canonical_sha256",
            }:
                raise ValueError("Invalid A1 block observation")
            shape = blocks["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or not all(type(n) is int and n > 0 for n in shape)
                or shape[1] != plan["seq_len"]
                or blocks["dtype"] != "int32"
                or blocks["byte_order"] != "little"
                or blocks["layout"] != "C"
                or type(blocks["byte_length"]) is not int
                or blocks["byte_length"] != 4 * shape[0] * shape[1]
            ):
                raise ValueError("Invalid A1 block dimensions")
            validate_id(blocks["canonical_sha256"])
            if evaluated:
                policy = next(
                    n for n in defs if n.name == "neural/a1/config/evaluation"
                )
                limit = cast(dict[str, Any], policy.to_dict()["payload"])["value"][
                    "eval_batches"
                ]
                batch = plan["batch_size"]
                size = min(shape[0], batch * limit)
                size = size // batch * batch or size
                prefix = d["evaluation_prefix"]
                if (
                    not isinstance(prefix, dict)
                    or not isinstance(prefix.get("canonical_sha256"), str)
                    or prefix
                    != {
                        **blocks,
                        "shape": [size, shape[1]],
                        "byte_length": 4 * size * shape[1],
                        "canonical_sha256": prefix["canonical_sha256"],
                    }
                    or (size == shape[0] and prefix != blocks)
                ):
                    raise ValueError("Invalid A1 evaluation prefix")
                validate_id(prefix["canonical_sha256"])
