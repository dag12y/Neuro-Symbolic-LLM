"""Portable student checkpoint provenance and offline verification."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .hashing import canonical_json, hash_directory, load_json
from .manifest import export_manifest, verify_manifest
from .records import ComponentRecord, Dependency
from .registry import Registry

_SCOPE = "directory tree v1"
_TRACKED = ".checkpoint-tracked"
_TRACKED_CONTENT = b"student-checkpoint-v1\n"
_PENDING = {"checkpoint_manifest_version": 1, "state": "saving"}


def checkpoint_manifest_path(directory: str | Path) -> Path:
    path = Path(directory).resolve()
    return path.with_name(path.name + ".checkpoint.versions.json")


def _training_digest(inputs: object) -> str:
    payload = b"versioning:student-training:v1\0" + canonical_json(inputs)
    return sha256(payload).hexdigest()


def register_training_provenance(
    registry: Registry,
    *,
    student_prompt_id: str,
    ontology_id: str,
    details: Mapping[str, object],
) -> str:
    """Register privacy-filtered run details and validated component dependencies."""
    inputs = dict(details)
    record = ComponentRecord(
        kind="student-training",
        name="semantic-student",
        content_digest=_training_digest(inputs),
        digest_scope="training settings and inputs v1",
        details=inputs,
        dependencies=(
            Dependency("student-prompt", student_prompt_id),
            Dependency("ontology", ontology_id),
        ),
    )
    _validate_training(
        record, registry.get(student_prompt_id), registry.get(ontology_id)
    )
    return registry.register(record)


def _validate_training(
    training: ComponentRecord, prompt: ComponentRecord, ontology: ComponentRecord
) -> None:
    if (
        training.kind != "student-training"
        or training.digest_scope != "training settings and inputs v1"
        or training.content_digest != _training_digest(training.to_dict()["details"])
        or prompt.kind != "student-prompt"
        or ontology.kind not in {"ontology-bundle", "ontology-catalog"}
        or training.dependencies != (
            Dependency("student-prompt", prompt.record_id),
            Dependency("ontology", ontology.record_id),
        )
    ):
        raise ValueError("Invalid student training provenance or component references")
    expected = (
        {dep.record_id for dep in ontology.dependencies}
        if ontology.kind == "ontology-bundle"
        else {ontology.record_id}
    )
    actual = {
        dep.record_id for dep in prompt.dependencies if dep.role != "semantic-contract"
    }
    if not expected or expected != actual:
        raise ValueError("Student prompt dependencies differ from training ontology")


def _write(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _lock(directory: Path, operation: int) -> Iterator[None]:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, operation)
        yield
    finally:
        os.close(fd)


@contextmanager
def checkpoint_save(directory: Path, training_manifest: bytes | None) -> Iterator[None]:
    """Serialize saves and invalidate prior success before an in-place overwrite."""
    path = checkpoint_manifest_path(directory)
    records = verify_manifest(training_manifest) if training_manifest else ()
    if training_manifest is not None:
        if not records:
            raise ValueError("Empty training manifest")
        _validate_training_root(records[-1], records)
    with _lock(directory, fcntl.LOCK_EX):
        if path.is_symlink():
            raise ValueError(f"Checkpoint manifest must not be a symlink: {path}")
        if path.exists():
            old = load_json(path.read_bytes())
            if old != _PENDING:
                # Preserve an unrelated file occupying the manifest location.
                prior = _records(old)[-1]
                if hash_directory(directory.resolve()) != prior.content_digest:
                    raise ValueError(
                        "Stale checkpoint manifest: existing directory differs"
                    )
        # Legacy or interrupted saves may contain externally supplied files.
        for parent, directories, files in os.walk(directory):
            for name in (*directories, *files):
                if (Path(parent) / name).is_symlink():
                    raise ValueError("Checkpoint directory contains a symlink")
        tracking = directory / _TRACKED
        if tracking.exists() and tracking.read_bytes() != _TRACKED_CONTENT:
            raise ValueError("Invalid checkpoint tracking flag")
        _write(path, canonical_json(_PENDING))
        if not tracking.exists():
            _write(tracking, _TRACKED_CONTENT)
        yield
        digest = hash_directory(directory.resolve())
        with TemporaryDirectory() as tmp:
            registry = Registry(tmp)
            for record in records:
                registry.register(record)
            checkpoint = ComponentRecord(
                kind="student-checkpoint",
                name="semantic-student",
                content_digest=digest,
                digest_scope=_SCOPE,
                dependencies=(
                    (Dependency("training", records[-1].record_id),) if records else ()
                ),
                details={} if records else {"training_provenance": "unavailable"},
            )
            registry.register(checkpoint)
            components = export_manifest(registry, checkpoint.record_id)
            value = {
                "checkpoint_manifest_version": 1,
                "components": load_json(components),
            }
            _write(path, canonical_json(value) + b"\n")


def _records(value: object) -> tuple[ComponentRecord, ...]:
    if (
        not isinstance(value, dict)
        or set(value) != {"checkpoint_manifest_version", "components"}
        or type(value["checkpoint_manifest_version"]) is not int
        or value["checkpoint_manifest_version"] != 1
    ):
        raise ValueError("Invalid checkpoint manifest format")
    records = verify_manifest(canonical_json(value["components"]))
    checkpoint = records[-1]
    if checkpoint.kind != "student-checkpoint" or checkpoint.digest_scope != _SCOPE:
        raise ValueError("Invalid checkpoint training dependency")
    if not checkpoint.dependencies:
        if checkpoint.details == {"training_provenance": "unavailable"}:
            return records
        raise ValueError("Checkpoint lacks training provenance or an unavailable status")
    if (
        len(checkpoint.dependencies) != 1
        or checkpoint.dependencies[0].role != "training"
    ):
        raise ValueError("Invalid checkpoint training dependency")
    # A checkpoint has one dependency, so the last record before the root
    # in M1 dependency order is its training record.
    _validate_training_root(records[-2], records)
    return records


def _validate_training_root(
    training: ComponentRecord, records: tuple[ComponentRecord, ...]
) -> None:
    indexed = {record.record_id: record for record in records}
    if len(training.dependencies) != 2:
        raise ValueError("Invalid student training component references")
    prompt, ontology = (indexed[dep.record_id] for dep in training.dependencies)
    _validate_training(training, prompt, ontology)


def verify_student_checkpoint(directory: str | Path) -> ComponentRecord | None:
    """Verify exact checkpoint bytes and graph. None means provenance unavailable."""
    root = Path(directory)
    path = checkpoint_manifest_path(root)
    with _lock(root, fcntl.LOCK_SH):
        if path.is_symlink():
            raise ValueError("Checkpoint manifest must not be a symlink")
        if not path.exists():
            tracking = root / _TRACKED
            if tracking.exists() or tracking.is_symlink():
                raise ValueError("Tracked checkpoint manifest is missing")
            return None
        value = load_json(path.read_bytes())
        if value == _PENDING:
            raise ValueError("Incomplete checkpoint save or manifest publication")
        record = _records(value)[-1]
        if hash_directory(root.resolve()) != record.content_digest:
            raise ValueError(f"Checkpoint directory SHA-256 mismatch: {root}")
        return record
