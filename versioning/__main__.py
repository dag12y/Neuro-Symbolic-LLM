"""Terminal access to the registry and offline version verifiers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .checkpoint import verify_student_checkpoint
from .export import export_sidecar_path, verify_export_sidecar
from .manifest import export_manifest, verify_manifest
from .records import ComponentRecord
from .registry import Registry

_LEGACY = 3


def _emit(args: argparse.Namespace, payload: object, message: str) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else message)


def _identity(record: ComponentRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "kind": record.kind,
        "name": record.name,
        "label": record.label,
        "label_role": record.label_role,
    }


def _describe(record: ComponentRecord) -> str:
    label = f" [{record.label_role}: {record.label}]" if record.label else ""
    return f"{record.record_id}  {record.kind}  {record.name}{label}"


def _registry(args: argparse.Namespace) -> Registry:
    registry = Registry(args.registry)
    if not registry.root.exists():
        raise FileNotFoundError(f"Registry not found: {registry.root}")
    if not registry.root.is_dir():
        raise NotADirectoryError(f"Registry is not a directory: {registry.root}")
    return registry


def _register_semantic(args: argparse.Namespace) -> None:
    # Keep the inspection and verification commands usable without PyYAML.
    from .semantic import register_semantic_components

    ids = register_semantic_components(
        Registry(args.registry), project_root=args.project_root
    )
    components = {
        "parser": ids.parser,
        "contract": ids.contract,
        "ontology_bundle": ids.ontology_bundle,
        "ontology_modules": dict(ids.ontology_modules),
        "prompts": dict(ids.prompts),
    }
    lines = [
        f"Parser root  {ids.parser}",
        f"Contract     {ids.contract}",
        f"Bundle       {ids.ontology_bundle}",
        *(
            f"Ontology {name}  {record_id}"
            for name, record_id in ids.ontology_modules.items()
        ),
        *(f"Prompt {name}  {record_id}" for name, record_id in ids.prompts.items()),
    ]
    _emit(args, components, "\n".join(lines))


def _list(args: argparse.Namespace) -> None:
    records = _registry(args).list_records(
        kind=args.kind, name=args.name, label=args.label
    )
    _emit(
        args,
        [_identity(record) for record in records],
        "\n".join(map(_describe, records)) or "No matching records.",
    )


def _inspect(args: argparse.Namespace) -> None:
    record = _registry(args).get(args.record_id)
    identity = {
        **_identity(record),
        "schema_version": record.schema_version,
        "canonicalization_version": record.canonicalization_version,
        "content_digest": record.content_digest,
        "digest_scope": record.digest_scope,
        "dependencies": [
            {"role": dependency.role, "record_id": dependency.record_id}
            for dependency in record.dependencies
        ],
    }
    lines = [
        _describe(record),
        f"Content SHA-256  {record.content_digest} ({record.digest_scope})",
        f"Schema {record.schema_version}, "
        f"canonicalization {record.canonicalization_version}",
        "Dependencies",
        *(f"  {dep.role}  {dep.record_id}" for dep in record.dependencies),
    ]
    if not record.dependencies:
        lines.append("  None")
    _emit(args, identity, "\n".join(lines))


def _export_manifest(args: argparse.Namespace) -> None:
    payload = export_manifest(_registry(args), args.record_id)
    if args.output is None:
        print(payload.decode("ascii"))
    else:
        with args.output.open("xb") as stream:
            stream.write(payload + b"\n")
        print(f"Manifest written to {args.output}")


def _verify_manifest(args: argparse.Namespace) -> None:
    records = verify_manifest(args.path.read_bytes())
    root_id = records[-1].record_id
    _emit(
        args,
        {"status": "verified", "root_id": root_id, "record_count": len(records)},
        f"Verified manifest  {root_id} ({len(records)} records)",
    )


def _artifact_verified(args: argparse.Namespace, record: ComponentRecord) -> None:
    _emit(
        args,
        {
            "status": "verified",
            "record_id": record.record_id,
            "content_digest": record.content_digest,
        },
        f"Verified {record.kind}  {record.record_id}\nSHA-256  {record.content_digest}",
    )


def _verify_export(args: argparse.Namespace) -> None:
    sidecar = (
        args.path
        if args.path.name.endswith(".versions.json")
        else export_sidecar_path(args.path)
    )
    _artifact_verified(args, verify_export_sidecar(sidecar))


def _verify_checkpoint(args: argparse.Namespace) -> int | None:
    record = verify_student_checkpoint(args.path)
    if record is None:
        _emit(
            args,
            {"status": "legacy", "message": "Checkpoint has no tracked provenance"},
            "Legacy or untracked checkpoint. Provenance is unavailable.",
        )
        return _LEGACY
    _artifact_verified(args, record)
    return None


def _status(args: argparse.Namespace) -> None:
    records = _registry(args).verify_integrity()
    kinds = Counter(record.kind for record in records)
    summary = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
    _emit(
        args,
        {"status": "verified", "record_count": len(records), "kinds": dict(kinds)},
        f"Registry integrity verified  ({len(records)} records)"
        + (f"\n{summary}" if summary else ""),
    )


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse can otherwise reflect arbitrary command-line values.
        self.print_usage(sys.stderr)
        self.exit(2, "Error: Invalid arguments. Use --help for command usage.\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="python -m versioning",
        description="Manage component provenance and verify local artifacts offline.",
        epilog=(
            "Exit codes: 0 success, 1 operation failed, 2 usage, "
            "3 untracked checkpoint."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text, handler, *, registry=False, structured=True):
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.set_defaults(handler=handler, json=False)
        if registry:
            command.add_argument(
                "--registry", type=Path, default=Path(".version-registry"),
                help="registry directory (default: .version-registry)",
            )
        if structured:
            command.add_argument("--json", action="store_true", help="output JSON")
        return command

    register = add(
        "register-semantic", "Register semantic parser components.", _register_semantic,
        registry=True,
    )
    register.add_argument(
        "--project-root", type=Path, default=Path("."),
        help="project checkout (default: current directory)",
    )
    listing = add("list", "List registered components.", _list, registry=True)
    for field in ("kind", "name", "label"):
        listing.add_argument(f"--{field}", help=f"match this {field} exactly")
    inspect = add(
        "inspect", "Inspect a component by exact ID.", _inspect, registry=True
    )
    inspect.add_argument("record_id", help="full SHA-256 record ID")
    export = add(
        "export-manifest", "Export a root and its dependency closure as JSON.",
        _export_manifest, registry=True, structured=False,
    )
    export.add_argument("record_id", help="full SHA-256 root record ID")
    export.add_argument(
        "--output", type=Path, help="new file (default: standard output)"
    )
    manifest = add(
        "verify-manifest", "Verify a portable component manifest.", _verify_manifest
    )
    manifest.add_argument("path", type=Path, help="manifest JSON file")
    parser_export = add(
        "verify-export", "Verify a parser export and its sidecar.", _verify_export
    )
    parser_export.add_argument(
        "path", type=Path, help="output JSONL or .versions.json sidecar"
    )
    checkpoint = add(
        "verify-checkpoint", "Verify a student checkpoint directory.", _verify_checkpoint
    )
    checkpoint.add_argument("path", type=Path, help="checkpoint directory")
    add(
        "status", "Verify the registry and summarize its records.", _status, registry=True
    )
    return parser


def _error_message(error: Exception) -> str:
    # Registry contents and parser configuration can appear in exception text.
    # Emit known diagnostics without their input-controlled values or paths.
    reason = str(error).partition(":")[0]
    if isinstance(error, FileNotFoundError):
        missing = {
            "Registry not found",
            "Version sidecar not found",
            "Export artifact not found",
        }
        return reason if reason in missing else "Missing file or directory"
    if isinstance(error, FileExistsError):
        return "Output file already exists"
    if isinstance(error, OSError):
        return "Cannot access the requested file or directory"
    if isinstance(error, ImportError):
        return "Semantic registration requires PyYAML in the Python environment"
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON at line {error.lineno}, column {error.colno}"
    safe_reasons = {
        "Artifact record does not link to the parser implementation",
        "Checkpoint directory SHA-256 mismatch",
        "Component record fields or identity do not match",
        "Conflicting release label",
        "Duplicate JSON key",
        "Duplicate manifest record",
        "Expected a lowercase SHA-256 hex digest",
        "Export artifact SHA-256 mismatch",
        "Incomplete checkpoint save or manifest publication",
        "Invalid checkpoint manifest format",
        "Invalid export version sidecar",
        "Invalid manifest fields",
        "Manifest contains unreachable records",
        "Missing dependency record",
        "Tracked checkpoint manifest is missing",
        "Unsupported manifest version",
    }
    if reason in safe_reasons:
        return reason
    if isinstance(error, SyntaxError):
        return "Invalid Python source in the semantic project"
    return "Invalid record, manifest, or semantic project configuration"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        return args.handler(args) or 0
    except (OSError, ValueError, ImportError, SyntaxError) as error:
        message = _error_message(error)
        if args.json:
            _emit(args, {"status": "error", "message": message}, "")
        else:
            print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
