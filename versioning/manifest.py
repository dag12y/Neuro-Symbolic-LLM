"""Portable Merkle closures with registry-independent verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .hashing import canonical_json, load_json
from .nodes import Node, reachable_nodes, validate_id

if TYPE_CHECKING:
    from .registry import Registry

MANIFEST_VERSION = 2


def export_manifest(registry: Registry, root_id: str) -> bytes:
    """Export exactly the root and its reachable dependencies in ID order."""
    validate_id(root_id)
    closure = reachable_nodes((root_id,), registry.get)
    nodes = sorted(closure, key=lambda node: node.id)
    return canonical_json(
        {
            "manifest_version": MANIFEST_VERSION,
            "root_id": root_id,
            "nodes": [node.to_dict() for node in nodes],
        }
    )


def verify_manifest(
    data: bytes | str, *, expected_root_id: str | None = None
) -> tuple[Node, ...]:
    """Verify graph identity offline and return nodes in dependency order."""
    manifest = load_json(data)
    fields = {"manifest_version", "root_id", "nodes"}
    if type(manifest) is not dict or set(manifest) != fields:
        raise ValueError("Invalid manifest fields")
    version = manifest["manifest_version"]
    if type(version) is not int or version != MANIFEST_VERSION:
        raise ValueError("Unsupported manifest version")
    root_id = manifest["root_id"]
    validate_id(root_id)
    if expected_root_id is not None:
        validate_id(expected_root_id)
        if root_id != expected_root_id:
            raise ValueError("Manifest root does not match the expected root ID")
    if type(manifest["nodes"]) is not list:
        raise ValueError("Manifest nodes must be an array")

    nodes: dict[str, Node] = {}
    for entry in manifest["nodes"]:
        node = Node.from_dict(entry)
        if node.id in nodes:
            raise ValueError(f"Duplicate manifest node: {node.id}")
        nodes[node.id] = node
    if list(nodes) != sorted(nodes):
        raise ValueError("Manifest nodes are out of ID order")
    ordered = reachable_nodes((root_id,), nodes.__getitem__)
    if len(ordered) != len(nodes):
        raise ValueError("Manifest contains unreachable nodes")
    if ordered[-1].name.startswith("workflow/"):
        from .lineage import validate_known

        validate_known(ordered)
    return ordered
