"""Command-line inspection, export, and offline graph verification."""

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from .hashing import canonical_json
from .manifest import export_manifest, verify_manifest
from .registry import Registry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m versioning", description="Inspect and verify Merkle graphs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="read a node by full ID")
    inspect.add_argument("--registry", required=True, type=Path)
    inspect.add_argument("node_id")

    export = commands.add_parser("export-manifest", help="write a closure to stdout")
    export.add_argument("--registry", required=True, type=Path)
    export.add_argument("root_id")

    verify = commands.add_parser("verify-manifest", help="verify a closure offline")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expected-root")

    snapshot = commands.add_parser(
        "snapshot-symbolic", help="register a symbolic definition from local files"
    )
    snapshot.add_argument("--project-root", type=Path, default=Path("."))
    snapshot.add_argument("--registry", required=True, type=Path)
    snapshot.add_argument("--recipe", choices=("m2a", "m2b"), default="m2b")
    snapshot.add_argument(
        "--ontology-bundle", type=Path, help="override the parser bundle"
    )

    a1 = commands.add_parser(
        "snapshot-a1",
        allow_abbrev=False,
        help="register an offline Stage A1 definition",
    )
    a1.add_argument(
        "--config", type=Path, default=Path("configs/stage_A/a1_lora_baseline.yaml")
    )
    a1.add_argument("--registry", type=Path, required=True)
    a1.add_argument("--rank", type=int)
    a1.add_argument("--epochs", type=int)
    a1.add_argument("--model")
    a1.add_argument("--eval-batches", type=int)
    a1.add_argument("--no-jit", action="store_true")

    adapter = commands.add_parser(
        "run-symbolic-export", help="convert parser results and record the export"
    )
    parser_export = commands.add_parser(
        "run-parser-export", help="parse documents and record the fresh export"
    )
    for export_command in (adapter, parser_export):
        export_command.add_argument("--input", required=True, type=Path)
        export_command.add_argument("--output-dir", required=True, type=Path)
        export_command.add_argument("--registry", required=True, type=Path)
        export_command.add_argument("--expected-definition")
    adapter.add_argument("--bundle-output", action="store_true")
    adapter.add_argument("--skip-unsupported-rules", action="store_true")
    parser_export.add_argument("--corpus-id")
    parser_export.add_argument("--include-metta", action="store_true")
    parser_export.add_argument("--asserted-only", action="store_true")
    parser_export.add_argument("--student-pairs", action="store_true")
    parser_export.add_argument("--require-structured-output", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "verify-manifest":
            nodes = verify_manifest(
                args.path.read_bytes(), expected_root_id=args.expected_root
            )
            print(f"Verified graph {nodes[-1].id} ({len(nodes)} nodes)")
        elif args.command == "inspect":
            node = Registry(args.registry).get(args.node_id)
            print(canonical_json(node.to_dict()).decode("ascii"))
        elif args.command == "snapshot-a1":
            from .neural import build_a1_definition

            nodes = build_a1_definition(
                args.config,
                model=args.model,
                rank=args.rank,
                epochs=args.epochs,
                eval_batches=args.eval_batches,
                no_jit=args.no_jit,
            )
            Registry(args.registry).register_batch(nodes)
            print(nodes[-1].id)
        elif args.command == "snapshot-symbolic":
            from .symbolic import snapshot_symbolic

            root = snapshot_symbolic(
                args.project_root,
                Registry(args.registry),
                recipe=args.recipe,
                ontology_bundle=args.ontology_bundle,
            )
            print(root.id)
        elif args.command in ("run-symbolic-export", "run-parser-export"):
            from .execution import record_parser_export_run, record_symbolic_export_run

            registry = Registry(args.registry)
            if args.command == "run-symbolic-export":
                result = record_symbolic_export_run(
                    args.input,
                    args.output_dir,
                    registry,
                    bundle_output=args.bundle_output,
                    skip_unsupported_rules=args.skip_unsupported_rules,
                    expected_definition=args.expected_definition,
                )
            else:
                result = record_parser_export_run(
                    args.input,
                    args.output_dir,
                    registry,
                    corpus_id=args.corpus_id,
                    include_metta=args.include_metta,
                    asserted_only=args.asserted_only,
                    student_pairs=args.student_pairs,
                    require_structured_output=args.require_structured_output,
                    expected_definition=args.expected_definition,
                )
            print(result.root.id)
            return 2 if result.rejected else 0
        else:
            manifest = export_manifest(Registry(args.registry), args.root_id)
            print(manifest.decode("ascii"))
    except KeyError as error:
        print(f"Error: node not found: {error.args[0]}", file=sys.stderr)
        return 1
    except (ImportError, OSError, ValueError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
