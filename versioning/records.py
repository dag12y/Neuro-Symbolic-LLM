"""Immutable component records and shared dependency validation."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from .hashing import CANONICALIZATION_VERSION, canonical_json, hash_identity

SCHEMA_VERSION = 1
LabelRole = Literal["release", "family", "locator", "none"]


def validate_digest(value: object) -> None:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 hex digest")


def _require_text(value: object, field_name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(map(_freeze, value))
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(map(_thaw, value))
    return value


def _freeze_object(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a JSON object")
    snapshot = _thaw(value) if isinstance(value, MappingProxyType) else dict(value)
    assert isinstance(snapshot, dict)
    canonical_json(snapshot)
    return MappingProxyType({key: _freeze(item) for key, item in snapshot.items()})


@dataclass(frozen=True, slots=True)
class Dependency:
    """A named relationship to an exact component record."""

    role: str
    record_id: str

    def __post_init__(self) -> None:
        _require_text(self.role, "dependency role")
        validate_digest(self.record_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentRecord:
    kind: str
    name: str
    content_digest: str
    digest_scope: str
    label: str | None = None
    label_role: LabelRole = "none"
    dependencies: tuple[Dependency, ...] = ()
    producer: Mapping[str, object] | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = field(init=False, default=SCHEMA_VERSION)
    canonicalization_version: int = field(init=False, default=CANONICALIZATION_VERSION)
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("kind", "name", "digest_scope"):
            _require_text(getattr(self, field_name), field_name)
        validate_digest(self.content_digest)
        if type(self.label_role) is not str or self.label_role not in (
            "release",
            "family",
            "locator",
            "none",
        ):
            raise ValueError("Invalid label role")
        if self.label_role == "none":
            if self.label is not None:
                raise ValueError("Label role 'none' requires an absent label")
        else:
            _require_text(self.label, "label")
        dependencies = tuple(self.dependencies)
        if any(type(item) is not Dependency for item in dependencies):
            raise ValueError("Expected Dependency instances")
        object.__setattr__(self, "dependencies", dependencies)
        if self.producer is not None:
            object.__setattr__(self, "producer", _freeze_object(self.producer))
        object.__setattr__(self, "details", _freeze_object(self.details))
        object.__setattr__(self, "record_id", hash_identity(self._identity()))

    def _identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonicalization_version": self.canonicalization_version,
            "kind": self.kind,
            "name": self.name,
            "label": self.label,
            "label_role": self.label_role,
            "content_digest": self.content_digest,
            "digest_scope": self.digest_scope,
            "dependencies": [
                {"role": dep.role, "record_id": dep.record_id}
                for dep in self.dependencies
            ],
            "producer": _thaw(self.producer),
            "details": _thaw(self.details),
        }

    def to_dict(self) -> dict[str, object]:
        """Return an independent JSON representation."""
        return {**self._identity(), "record_id": self.record_id}

    @classmethod
    def from_dict(cls, value: object) -> "ComponentRecord":
        """Validate a serialized record against its recomputed identity."""
        if type(value) is not dict:
            raise ValueError("Expected a component record object")
        fields = value.copy()
        for key in ("record_id", "schema_version", "canonicalization_version"):
            fields.pop(key, None)
        try:
            fields["dependencies"] = tuple(
                Dependency(**dep) for dep in fields["dependencies"]
            )
            record = cls(**fields)
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid component record fields") from exc
        if canonical_json(record.to_dict()) != canonical_json(value):
            raise ValueError("Component record fields or identity do not match")
        return record


def dependency_order(
    root_id: str, records: Mapping[str, ComponentRecord]
) -> list[ComponentRecord]:
    """Validate the reachable graph and return each dependency before its users."""
    ordered: list[ComponentRecord] = []
    active: set[str] = set()
    visited: set[str] = set()
    pending = [(root_id, False)]
    while pending:
        record_id, expanded = pending.pop()
        if record_id in visited:
            continue
        if record_id not in records:
            raise ValueError(f"Missing dependency record: {record_id}")
        record = records[record_id]
        if expanded:
            active.remove(record_id)
            visited.add(record_id)
            ordered.append(record)
        else:
            if record_id in active:
                raise ValueError(f"Dependency cycle at {record_id}")
            active.add(record_id)
            pending.append((record_id, True))
            pending.extend(
                (dep.record_id, False) for dep in reversed(record.dependencies)
            )
    return ordered


def check_release_labels(records: Iterable[ComponentRecord]) -> None:
    releases: dict[tuple[str, str, str | None], str] = {}
    for record in records:
        if record.label_role == "release":
            key = (record.kind, record.name, record.label)
            if releases.setdefault(key, record.record_id) != record.record_id:
                raise ValueError(f"Conflicting release label: {key}")
