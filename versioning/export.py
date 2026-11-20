"""Publish and verify version sidecars for finalized parser export artifacts."""

import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

from .hashing import canonical_json, hash_file, load_json
from .manifest import verify_manifest
from .records import ComponentRecord, Dependency, validate_digest

EXPORT_MANIFEST_VERSION = 1
_SUFFIX = ".versions.json"
_ARTIFACT_KIND = "parser-export-artifact"
_DIGEST_SCOPE = "raw semantic document JSONL bytes"


def export_sidecar_path(output_path: str | Path) -> Path:
    """Return the deterministic version sidecar path for an export."""
    output = Path(output_path)
    if (
        output.name in ("", ".", "..")
        or any(char in output.name for char in ("\\", "\0"))
    ):
        raise ValueError("Export artifact requires a portable basename")
    return output.with_name(output.name + _SUFFIX)


def publish_export_sidecar(
    output: Path,
    *,
    expected_sha256: str,
    source_sha256: str,
    parser_record_id: str,
    component_manifest: bytes,
    settings: Mapping[str, object],
    unresolved: list[str],
) -> Path:
    """Publish a finalized output's sidecar, or recognize an identical retry."""
    path = export_sidecar_path(output)
    parser = verify_manifest(component_manifest)[-1]
    if parser.kind != "parser-implementation" or parser.record_id != parser_record_id:
        raise ValueError("Component manifest does not match the parser implementation")
    if output.is_symlink() or not output.is_file():
        raise ValueError(f"Export artifact must be a regular file: {output}")
    with output.open("rb") as stream:
        os.fsync(stream.fileno())
    actual = hash_file(output)
    if actual != expected_sha256:
        raise ValueError(f"Finalized export artifact SHA-256 mismatch: {output}")
    validate_digest(source_sha256)
    artifact = ComponentRecord(
        kind=_ARTIFACT_KIND,
        name=output.name,
        content_digest=actual,
        digest_scope=_DIGEST_SCOPE,
        dependencies=(Dependency("parser-implementation", parser_record_id),),
        details={
            "source_sha256": source_sha256,
            "settings": dict(settings),
            "unresolved": unresolved,
        },
    )
    data = canonical_json(
        {
            "export_manifest_version": EXPORT_MANIFEST_VERSION,
            "artifact": artifact.to_dict(),
            "components": load_json(component_manifest),
        }
    ) + b"\n"
    if path.is_symlink() or (
        path.exists() and (not path.is_file() or path.read_bytes() != data)
    ):
        raise FileExistsError(f"Conflicting version sidecar: {path}")
    if not path.exists():
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", delete=False
            ) as staged:
                temporary = Path(staged.name)
                staged.write(data)
                staged.flush()
                os.fsync(staged.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def verify_export_sidecar(sidecar_path: str | Path) -> ComponentRecord:
    """Verify sidecar and adjacent output offline and return the artifact record.

    Missing files raise FileNotFoundError. Invalid provenance or bytes raise
    ValueError. A valid result establishes integrity, with no authenticity claim.
    """
    path = Path(sidecar_path)
    if path.is_symlink():
        raise ValueError(f"Version sidecar must not be a symlink: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Version sidecar not found: {path}")
    if not path.is_file():
        raise ValueError(f"Version sidecar must be a regular file: {path}")
    try:
        value = load_json(path.read_bytes())
        if (
            type(value) is not dict
            or set(value) != {"export_manifest_version", "artifact", "components"}
            or type(value["export_manifest_version"]) is not int
            or value["export_manifest_version"] != EXPORT_MANIFEST_VERSION
        ):
            raise ValueError("Invalid export manifest fields or version")
        artifact = ComponentRecord.from_dict(value["artifact"])
        parser = verify_manifest(canonical_json(value["components"]))[-1]
        if (
            artifact.kind != _ARTIFACT_KIND
            or artifact.digest_scope != _DIGEST_SCOPE
            or artifact.label_role != "none"
            or parser.kind != "parser-implementation"
            or artifact.dependencies
            != (Dependency("parser-implementation", parser.record_id),)
        ):
            raise ValueError("Artifact record does not link to the parser implementation")
        locator = artifact.name
        if (
            locator in (".", "..")
            or any(char in locator for char in ("/", "\\", "\0"))
            or path.name != locator + _SUFFIX
        ):
            raise ValueError("Invalid or mismatched artifact locator")
        details = artifact.details
        unresolved = details.get("unresolved")
        if (
            set(details) != {"source_sha256", "settings", "unresolved"}
            or not isinstance(details["settings"], Mapping)
            or not isinstance(unresolved, tuple)
            or any(type(item) is not str for item in unresolved)
        ):
            raise ValueError("Invalid export artifact details")
        validate_digest(details["source_sha256"])
    except ValueError as error:
        raise ValueError(f"Invalid export version sidecar {path}: {error}") from error
    output = path.with_name(locator)
    if output.is_symlink():
        raise ValueError(f"Export artifact must not be a symlink: {output}")
    if not output.exists():
        raise FileNotFoundError(f"Export artifact not found: {output}")
    if not output.is_file():
        raise ValueError(f"Export artifact must be a regular file: {output}")
    if hash_file(output) != artifact.content_digest:
        raise ValueError(f"Export artifact SHA-256 mismatch: {output}")
    return artifact
