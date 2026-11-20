"""Portable snapshots of a component record and its dependency graph."""

from .hashing import canonical_json, load_json
from .records import (
    ComponentRecord,
    check_release_labels,
    dependency_order,
    validate_digest,
)
from .registry import Registry

MANIFEST_VERSION = 1


def export_manifest(registry: Registry, root_id: str) -> bytes:
    """Export the complete reachable graph as canonical JSON bytes."""
    records: dict[str, ComponentRecord] = {}
    pending = [root_id]
    while pending:
        record_id = pending.pop()
        if record_id not in records:
            record = registry.get(record_id)
            records[record_id] = record
            pending.extend(dep.record_id for dep in record.dependencies)
    dependency_order(root_id, records)
    check_release_labels(records.values())
    return canonical_json(
        {
            "manifest_version": MANIFEST_VERSION,
            "root_id": root_id,
            "records": [records[key].to_dict() for key in sorted(records)],
        }
    )


def verify_manifest(data: bytes | str) -> tuple[ComponentRecord, ...]:
    """Verify offline and return records in dependency order, with the root last."""
    value = load_json(data)
    if type(value) is not dict or set(value) != {
        "manifest_version",
        "root_id",
        "records",
    }:
        raise ValueError("Invalid manifest fields")
    version = value["manifest_version"]
    if type(version) is not int or version != MANIFEST_VERSION:
        raise ValueError("Unsupported manifest version")
    root_id = value["root_id"]
    validate_digest(root_id)
    if type(value["records"]) is not list:
        raise ValueError("Manifest records must be an array")
    records: dict[str, ComponentRecord] = {}
    for item in value["records"]:
        record = ComponentRecord.from_dict(item)
        if record.record_id in records:
            raise ValueError(f"Duplicate manifest record: {record.record_id}")
        records[record.record_id] = record
    ordered = dependency_order(root_id, records)
    if len(ordered) != len(records):
        raise ValueError("Manifest contains unreachable records")
    check_release_labels(ordered)
    return tuple(ordered)
