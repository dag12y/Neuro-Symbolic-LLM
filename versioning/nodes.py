"""Immutable Merkle nodes and dependency validation."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar

from .hashing import CANONICALIZATION_VERSION, canonical_json, hash_identity

SCHEMA_VERSION = 1
_KINDS = frozenset(
    {"source", "external_release", "composite", "semantic_content", "artifact"}
)
_ROLE_TARGET = {
    "uses-source": "source",
    "uses-release": "external_release",
    "uses-component": "composite",
    "uses-semantic": "semantic_content",
    "uses-artifact": "artifact",
    "produced-by": "composite",
}
_NODE_FIELDS = {
    "schema_version",
    "canonicalization_version",
    "kind",
    "name",
    "payload",
    "dependencies",
    "id",
}
_EDGE_FIELDS = {"role", "slot", "target_id"}
_GIT_PATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*")
_PACKAGE_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


def validate_id(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 hex digest")


def _logical_name(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.isprintable()
        or value.strip() != value
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"{field_name} must be a relative logical name")


def _require_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(expected))}")
    return value


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


def _validate_release(name: str, payload: dict[str, Any]) -> None:
    _require_fields(payload, {"repository", "selector"}, "Release payload")
    if not _PACKAGE_NAME.fullmatch(name):
        raise ValueError("Release name must be a normalized package name")
    repository = payload["repository"]
    if not isinstance(repository, str) or not repository.startswith("https://"):
        raise ValueError("Release repository must be a normalized HTTPS Git URL")
    host, _, path = repository.removeprefix("https://").partition("/")
    if not _GIT_PATH.fullmatch(path) or not all(
        _HOST_LABEL.fullmatch(label) for label in host.split(".")
    ):
        raise ValueError("Release repository must be a normalized HTTPS Git URL")

    selector = _require_fields(
        payload["selector"], {"kind", "value"}, "Release selector"
    )
    value = selector["value"]
    match selector["kind"]:
        case "tag":
            if (
                not isinstance(value, str)
                or not _GIT_PATH.fullmatch(value)
                or value.endswith((".", ".lock"))
                or ".." in value
                or ".lock/" in value
            ):
                raise ValueError("Release tag must be a Git tag name")
        case "commit":
            if not isinstance(value, str) or not re.fullmatch(
                "[0-9a-f]{40}|[0-9a-f]{64}", value
            ):
                raise ValueError("Expected a full lowercase Git object ID")
        case _:
            raise ValueError("Release selector kind must be tag or commit")


def _validate_payload(kind: str, name: str, payload: dict[str, Any]) -> None:
    if kind in {"source", "artifact"}:
        _require_fields(payload, {"sha256", "byte_length"}, f"{kind.title()} payload")
        validate_id(payload["sha256"])
        length = payload["byte_length"]
        if type(length) is not int or length < 0:
            raise ValueError(
                f"{kind.title()} byte_length must be a nonnegative integer"
            )
    elif kind == "external_release":
        _validate_release(name, payload)
    elif kind == "semantic_content":
        _require_fields(payload, {"projection", "version", "value"}, "Semantic payload")
        _logical_name(payload["projection"], "Projection name")
        version = payload["version"]
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("Projection version must be a positive integer")
        if not isinstance(payload["value"], dict):
            raise ValueError("Semantic value must be a JSON object")
    else:
        _require_fields(payload, {"recipe", "purpose"}, "Composite payload")
        recipe = _require_fields(payload["recipe"], {"name", "version"}, "Recipe")
        _logical_name(recipe["name"], "Recipe name")
        if type(recipe["version"]) is not int or recipe["version"] < 1:
            raise ValueError("Recipe version must be a positive integer")
        if payload["purpose"] not in ("component", "definition", "run"):
            raise ValueError("Composite purpose must be component, definition, or run")


def _validate_slot(role: str, slot: str) -> None:
    if not isinstance(role, str) or role not in _ROLE_TARGET:
        raise ValueError(f"Unsupported dependency role: {role!r}")
    _logical_name(slot, "Dependency slot")


@dataclass(frozen=True, slots=True)
class Dependency:
    role: str
    slot: str
    target_id: str

    def __post_init__(self) -> None:
        _validate_slot(self.role, self.slot)
        validate_id(self.target_id)


@dataclass(frozen=True, slots=True)
class DraftDependency:
    role: str
    slot: str
    target: str

    def __post_init__(self) -> None:
        _validate_slot(self.role, self.slot)
        _logical_name(self.target, "Draft target")


_Edge = TypeVar("_Edge", Dependency, DraftDependency)


def _ordered_edges(edges: Iterable[_Edge], edge_type: type[_Edge]) -> tuple[_Edge, ...]:
    edges = tuple(edges)
    if any(type(edge) is not edge_type for edge in edges):
        raise ValueError(f"Expected {edge_type.__name__} instances")
    ordered = tuple(sorted(edges, key=lambda edge: (edge.role, edge.slot)))
    slots = set()
    for edge in ordered:
        key = (edge.role, edge.slot)
        if key in slots:
            raise ValueError(f"Duplicate dependency slot: {edge.role}/{edge.slot}")
        slots.add(key)
    return ordered


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    kind: str
    name: str
    payload: Mapping[str, object]
    dependencies: tuple[Dependency, ...] = ()
    schema_version: int = field(init=False, default=SCHEMA_VERSION)
    canonicalization_version: int = field(init=False, default=CANONICALIZATION_VERSION)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _KINDS:
            raise ValueError(f"Unsupported node kind: {self.kind!r}")
        _logical_name(self.name, "Node name")
        if not isinstance(self.payload, Mapping):
            raise ValueError("Node payload must be a JSON object")
        payload = dict(self.payload)
        canonical_json(payload)
        _validate_payload(self.kind, self.name, payload)

        dependencies = _ordered_edges(self.dependencies, Dependency)
        if self.kind in {"source", "external_release"} and dependencies:
            raise ValueError(f"{self.kind} nodes cannot have dependencies")
        if self.kind == "composite" and not dependencies:
            raise ValueError("Composite nodes require at least one dependency")
        if self.kind == "artifact" and (
            len(dependencies) != 1
            or (dependencies[0].role, dependencies[0].slot) != ("produced-by", "inputs")
        ):
            raise ValueError(
                "Artifacts require one produced-by edge in the inputs slot"
            )

        object.__setattr__(self, "payload", _freeze(payload))
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "id", hash_identity(self._identity()))

    def _identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonicalization_version": self.canonicalization_version,
            "kind": self.kind,
            "name": self.name,
            "payload": _thaw(self.payload),
            "dependencies": [
                {"role": edge.role, "slot": edge.slot, "target_id": edge.target_id}
                for edge in self.dependencies
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity(), "id": self.id}

    @classmethod
    def from_dict(cls, value: object) -> Node:
        fields = _require_fields(value, _NODE_FIELDS, "Node")
        for key, version in (
            ("schema_version", SCHEMA_VERSION),
            ("canonicalization_version", CANONICALIZATION_VERSION),
        ):
            if type(fields[key]) is not int or fields[key] != version:
                raise ValueError(f"Unsupported node {key}")
        validate_id(fields["id"])
        if type(fields["dependencies"]) is not list:
            raise ValueError("Node dependencies must be an array")
        edges = tuple(
            Dependency(**_require_fields(edge, _EDGE_FIELDS, "Dependency"))
            for edge in fields["dependencies"]
        )
        node = cls(
            kind=fields["kind"],
            name=fields["name"],
            payload=fields["payload"],
            dependencies=edges,
        )
        if edges != node.dependencies:
            raise ValueError("Node dependencies are out of canonical order")
        if node.id != fields["id"]:
            raise ValueError(f"Node ID does not match its identity: {fields['id']}")
        return node


@dataclass(frozen=True, slots=True, kw_only=True)
class DraftNode:
    """A node specification whose dependencies refer to names in the same draft."""

    kind: str
    name: str
    payload: Mapping[str, object]
    dependencies: tuple[DraftDependency, ...] = ()


def dependency_order(
    roots: Iterable[str],
    edges_for: Callable[[str], Iterable[tuple[str, str]]],
) -> tuple[str, ...]:
    """Visit (slot, target) edges iteratively, returning dependencies first."""
    ordered: list[str] = []
    visited: set[str] = set()

    def outgoing(key: str) -> Iterable[tuple[str, str]]:
        try:
            return edges_for(key)
        except KeyError as error:
            raise ValueError(f"Missing dependency node: {key}") from error

    for root in roots:
        if root in visited:
            continue
        active = {root: 0}
        stack = [(root, iter(outgoing(root)))]
        while stack:
            parent, edges = stack[-1]
            edge = next(edges, None)
            if edge is None:
                stack.pop()
                del active[parent]
                visited.add(parent)
                ordered.append(parent)
                continue
            slot, target = edge
            if target in visited:
                continue
            if target in active:
                path = [key for key, _ in stack[active[target] :]]
                path.append(target)
                cycle = " -> ".join(path)
                raise ValueError(f"Dependency cycle at slot {slot}: {cycle}")
            active[target] = len(stack)
            stack.append((target, iter(outgoing(target))))
    return tuple(ordered)


def _validate_target(edge: Dependency, target: Node) -> None:
    expected = _ROLE_TARGET[edge.role]
    if target.kind != expected:
        raise ValueError(
            f"Dependency {edge.role}/{edge.slot} requires {expected}, got {target.kind}"
        )


def build_nodes(drafts: Mapping[str, DraftNode]) -> dict[str, Node]:
    """Resolve and hash named drafts bottom-up, reusing shared children."""
    for key in drafts:
        _logical_name(key, "Draft key")
    edges_by_key = {
        key: _ordered_edges(draft.dependencies, DraftDependency)
        for key, draft in drafts.items()
    }

    def edges_for(key: str) -> tuple[tuple[str, str], ...]:
        return tuple((edge.slot, edge.target) for edge in edges_by_key[key])

    nodes: dict[str, Node] = {}
    for key in dependency_order(sorted(drafts), edges_for):
        draft = drafts[key]
        dependencies = []
        for edge in edges_by_key[key]:
            target = nodes[edge.target]
            dependency = Dependency(edge.role, edge.slot, target.id)
            _validate_target(dependency, target)
            dependencies.append(dependency)
        nodes[key] = Node(
            kind=draft.kind,
            name=draft.name,
            payload=draft.payload,
            dependencies=tuple(dependencies),
        )
    return nodes


def reachable_nodes(
    roots: Iterable[str], lookup: Callable[[str], Node]
) -> tuple[Node, ...]:
    """Validate resolved node references and return the closure dependencies first."""
    nodes: dict[str, Node] = {}

    def edges_for(key: str) -> tuple[tuple[str, str], ...]:
        node = lookup(key)
        if node.id != key:
            raise ValueError(f"Dependency lookup returned the wrong node: {key}")
        nodes[key] = node
        return tuple((edge.slot, edge.target_id) for edge in node.dependencies)

    ordered = tuple(nodes[key] for key in dependency_order(roots, edges_for))
    for node in ordered:
        for edge in node.dependencies:
            _validate_target(edge, nodes[edge.target_id])
    return ordered
