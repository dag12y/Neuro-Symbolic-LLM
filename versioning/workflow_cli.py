"""CLI wiring for the observed stages and byte-checked lineage records."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

RUNS = {
    "run-miner": "miner",
    "run-embedding": "embedding",
    "run-a1": "a1-train",
    "evaluate-a1": "a1-eval",
    "train-student": "student-train",
    "evaluate-student": "student-eval",
}
COMMANDS = {
    "inspect-graph",
    "verify-output",
    *RUNS,
    "resolve-miner",
    "link-runs",
    "verify-file",
    "import-file",
}


def add_commands(commands: Any) -> None:
    for name, help_text in {
        "run-miner": "run the five v0.1.1 source-grounded mining commands",
        "run-embedding": "encode source spans from semantic documents and pattern records",
        "run-a1": "train with the existing Stage A1 driver",
        "evaluate-a1": "evaluate actual checkpoint bytes with Stage A1",
        "train-student": "train the semantic parser student on JSONL pairs",
        "evaluate-student": "evaluate a local student model on one source per line",
        "resolve-miner": "observe the selected miner release",
        "link-runs": "verify typed producer outputs against captured consumer inputs",
        "inspect-graph": "write manifest dependency edges as Graphviz DOT",
        "verify-output": "compare a named run output against supplied bytes",
        "verify-file": "compare a file with a source or artifact in a manifest",
        "import-file": "register external bytes with unknown authorship",
    }.items():
        cmd = commands.add_parser(name, help=help_text)
        if name not in ("verify-file", "verify-output", "inspect-graph"):
            cmd.add_argument("--registry", required=True, type=Path)
        if name in RUNS:
            cmd.add_argument("--output-dir", required=True, type=Path)
        if name in ("resolve-miner", "run-miner"):
            cmd.add_argument(
                "--source", type=Path, help="trusted hybrid-miner Git checkout"
            )
        if name == "resolve-miner":
            cmd.add_argument(
                "--installed",
                action="store_true",
                help="compare installed package files with --source",
            )
            cmd.add_argument(
                "--remote", action="store_true", help="look up tag at declared origin"
            )
        if name in ("run-miner", "run-embedding"):
            cmd.add_argument("--documents", required=True, type=Path)
        if name == "run-miner":
            for flag in ("bundle", "petta-root", "miner-root"):
                cmd.add_argument("--" + flag, required=True, type=Path)
            cmd.add_argument("--min-support", type=int, default=2)
        if name == "run-embedding":
            cmd.add_argument("--patterns", required=True, type=Path)
            cmd.add_argument("--model", default="gpt2")
            for key, default in (
                ("symbolic-dim", 256),
                ("max-length", 256),
                ("projection-seed", 42),
                ("layer-idx", None),
            ):
                cmd.add_argument("--" + key, type=int, default=default)
        if name in ("run-a1", "evaluate-a1"):
            cmd.add_argument(
                "--config",
                type=Path,
                default=Path("configs/stage_A/a1_lora_baseline.yaml"),
            )
        if name == "evaluate-a1":
            cmd.add_argument("--checkpoint", required=True, type=Path)
            cmd.add_argument("--split", default="test")
        if name == "train-student":
            cmd.add_argument(
                "--config",
                type=Path,
                default=Path("configs/parser_config/student_config.yaml"),
            )
            cmd.add_argument("--train-file", type=Path, required=True)
            cmd.add_argument("--val-file", type=Path)
            cmd.add_argument("--method", choices=["full", "lora"], default="full")
        if name == "evaluate-student":
            cmd.add_argument("--sentences", required=True, type=Path)
            cmd.add_argument("--model-dir", required=True, type=Path)
        if name == "link-runs":
            cmd.add_argument("consumer")
            cmd.add_argument(
                "--from-run", metavar="INPUT=ROOT", action="append", required=True
            )
        if name == "import-file":
            cmd.add_argument("file", type=Path)
            cmd.add_argument(
                "--kind",
                required=True,
                choices=["parser-results", "documents", "patterns", "checkpoint"],
            )
        if name == "inspect-graph":
            cmd.add_argument("manifest", type=Path)
            cmd.add_argument("--expected-root")
        if name in ("verify-file", "verify-output"):
            cmd.add_argument("manifest", type=Path)
            cmd.add_argument("node_id" if name == "verify-file" else "slot")
            cmd.add_argument("file", type=Path)
            cmd.add_argument("--expected-root")


def execute(args: argparse.Namespace) -> int:
    from .execution import _file_identity
    from .lineage import link, verify_file
    from .manifest import verify_manifest
    from .mining import mine, release
    from .nodes import Node
    from .registry import Registry
    from .workflow import capture
    from .workflow_stages import a1, embed, stage_definition, student, validate_options

    if args.command in ("verify-file", "verify-output", "inspect-graph"):
        nodes = verify_manifest(
            args.manifest.read_bytes(), expected_root_id=args.expected_root
        )
        from .lineage import underlying, validate_known
        from .workflow import slots

        validate_known(nodes)
        if args.command == "inspect-graph":
            import json

            print("digraph provenance {")
            for n in nodes:
                print(f'  "{n.id}" [label={json.dumps(n.name)}];')
                for e in n.dependencies:
                    print(
                        f'  "{n.id}" -> "{e.target_id}" [label={json.dumps(e.role + " " + e.slot)}];'
                    )
            print("}")
            return 0
        if args.command == "verify-output":
            index = {n.id: n for n in nodes}
            root = underlying(nodes[-1], index)
            out = slots(root, index).get(args.slot)
            if out is None or out.kind != "artifact":
                raise ValueError("Unknown output slot")
            verify_file(nodes, out.id, args.file)
        else:
            verify_file(nodes, args.node_id, args.file)
        print(nodes[-1].id)
        return 0
    reg = Registry(args.registry)
    if args.command == "import-file":
        node = Node(
            kind="source", name=f"import/{args.kind}", payload=_file_identity(args.file)
        )
        reg.register_batch([node])
        print(node.id)
        return 0
    if args.command == "link-runs":
        pairs = [v.split("=", 1) for v in args.from_run]
        if any(len(p) != 2 for p in pairs) or len(dict(pairs)) != len(pairs):
            raise ValueError("Expected unique INPUT=ROOT bindings")
        print(link(reg, args.consumer, dict(pairs)).id)
        return 0
    if args.command == "resolve-miner":
        if args.installed:
            from .mining import installed_release

            if args.source is None or args.remote:
                raise ValueError("--installed requires --source")
            nodes = installed_release(args.source)
        else:
            if args.source is not None and args.remote:
                raise ValueError("Select local or remote resolution")
            nodes = release(args.source, args.remote)
        reg.register_batch(nodes)
        print(nodes[-1].id)
        return 0
    stage = RUNS[args.command]
    from .workflow_stages import SETTINGS

    opts = {k: getattr(args, k) for k in SETTINGS[stage]}
    validate_options(stage, opts)
    files: dict[str, Path] = {}

    def definition() -> tuple[Node, ...]:
        if stage == "miner":
            return release(args.source)
        bundle = None
        if stage == "student-train":
            from .yaml_inputs import read_yaml

            _, values = read_yaml(args.config, Path.read_bytes)
            bundle = Path(values["data"]["ontology_bundle_manifest"]).resolve()
        return stage_definition(stage, getattr(args, "config", None), bundle)

    if stage in ("embedding", "miner"):
        files["documents.jsonl"] = args.documents
        if stage == "miner":
            files["bundle.json"] = args.bundle
        else:
            files["patterns.jsonl"] = args.patterns
    if stage in ("student-train", "a1-train", "a1-eval"):
        files["config.yaml"] = args.config
    if stage == "a1-eval":
        from .neural_integrity import resolve_checkpoint_path

        files["checkpoint.npz"] = resolve_checkpoint_path(args.checkpoint)
    if stage == "student-train":
        files["train.jsonl"] = args.train_file
        if args.val_file:
            files["validation.jsonl"] = args.val_file
    if stage == "student-eval":
        files["sentences.txt"] = args.sentences
        for p in args.model_dir.rglob("*"):
            if p.is_symlink() or p.suffix in (".bin", ".pt", ".pth", ".pkl", ".py"):
                raise ValueError("Expected a local safetensors student model")
            if p.is_file():
                files["model/" + p.relative_to(args.model_dir).as_posix()] = p
        if not any(k.endswith(".safetensors") for k in files):
            raise ValueError("Missing student safetensors weights")
    selected = definition()

    def produce(private: Path, out: Path, runtime: dict[str, Any]) -> None:
        if stage == "miner":
            resolution = cast(dict[str, Any], selected[1].to_dict()["payload"])["value"]
            if not args.source or not resolution["commit"]:
                raise ValueError("Miner source unavailable")
            mine(
                private,
                out,
                runtime,
                repo=args.source,
                commit=resolution["commit"],
                petta=args.petta_root,
                miner=args.miner_root,
                support=args.min_support,
            )
        else:
            handler = (
                embed
                if stage == "embedding"
                else a1
                if stage.startswith("a1-")
                else student
            )
            handler(private, out, runtime, opts)

    def checked_definition() -> tuple[Node, ...]:
        result = definition()
        if selected != result:
            raise ValueError("Definition changed")
        return result

    root, ok = capture(
        stage, reg, files, args.output_dir, checked_definition, opts, produce
    )
    print(root.id)
    return 0 if ok else 1
