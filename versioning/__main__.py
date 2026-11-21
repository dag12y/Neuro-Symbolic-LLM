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
        elif args.command == "snapshot-symbolic":
            from .symbolic import snapshot_symbolic

            root = snapshot_symbolic(
                args.project_root,
                Registry(args.registry),
                recipe=args.recipe,
                ontology_bundle=args.ontology_bundle,
            )
            print(root.id)
        else:
            manifest = export_manifest(Registry(args.registry), args.root_id)
            print(manifest.decode("ascii"))
    except KeyError as error:
        print(f"Error: node not found: {error.args[0]}", file=sys.stderr)
        return 1
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
