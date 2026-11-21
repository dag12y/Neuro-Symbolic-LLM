"""Capture the limited M2a symbolic definition from local project inputs."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from .nodes import Dependency, Node, reachable_nodes

if TYPE_CHECKING:
    from .registry import Registry

_MINER = "hybrid-miner"
_REPOSITORY = "https://github.com/iCog-Labs-Dev/hybrid-miner.git"
_PROJECT_SECTION = "[project.optional-dependencies].miner"
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_COMMIT = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
_VERSION_TAG = re.compile(r"v?[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9][A-Za-z0-9.-]*)?")

_PARSER_CONTRACT = "parser/semantic/schema.py"
_ROOT = "symbolic/definition"
# The recipes are listed with component definitions before their consumer.
_REQUIRED_SLOTS = {
    "symbolic/parser": {
        ("uses-source", "implementation"): "parser/semantic/semantic_parser.py",
        ("uses-source", "parser-contract"): _PARSER_CONTRACT,
    },
    "symbolic/adapter": {
        ("uses-source", "conversion"): "symbolic_pipeline/miner_adapter.py",
        ("uses-source", "bundle-compiler"): "symbolic_pipeline/miner_bundle.py",
        ("uses-source", "document-export"): "symbolic_pipeline/document_export.py",
        ("uses-source", "parser-contract"): _PARSER_CONTRACT,
        ("uses-release", "hybrid-miner"): _MINER,
    },
    "symbolic/mining": {
        ("uses-release", "hybrid-miner"): _MINER,
    },
    _ROOT: {
        ("uses-component", "parser"): "symbolic/parser",
        ("uses-component", "adapter"): "symbolic/adapter",
        ("uses-component", "mining"): "symbolic/mining",
    },
}


def _is_miner_requirement(requirement: str) -> bool:
    name = _PACKAGE_NAME.match(requirement.lstrip())
    return name is not None and re.sub(r"[-_.]+", "-", name[0].lower()) == _MINER


def _miner_requirement(pyproject: Path) -> str:
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    project = config.get("project")
    extras = project.get("optional-dependencies") if isinstance(project, dict) else None
    requirements = extras.get("miner") if isinstance(extras, dict) else None
    if requirements is None:
        raise ValueError(f"Missing {_PROJECT_SECTION}")
    if not isinstance(requirements, list):
        raise ValueError(f"{_PROJECT_SECTION} must be an array of requirement strings")
    matches: list[str] = []
    for requirement in requirements:
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError(f"{_PROJECT_SECTION} must contain requirement strings")
        if _is_miner_requirement(requirement):
            matches.append(requirement)
    if not matches:
        raise ValueError(f"Missing {_MINER} requirement in {_PROJECT_SECTION}")
    if len(matches) != 1:
        raise ValueError(f"Duplicate {_MINER} requirements in {_PROJECT_SECTION}")
    return matches[0]


def read_hybrid_miner_release(pyproject: str | Path) -> Node:
    """Read the declared release leaf without importing or resolving the miner."""
    requirement = _miner_requirement(Path(pyproject))
    name, separator, reference = requirement.partition("@")
    if not separator or re.sub(r"[-_.]+", "-", name.strip().lower()) != _MINER:
        raise ValueError(f"{_MINER} must be an exact Git direct requirement")
    reference = reference.strip()
    if not reference.lower().startswith("git+https://"):
        raise ValueError(f"{_MINER} requires a git+https repository URL")

    repository, separator, selector = reference[4:].rpartition("@")
    if not separator:
        raise ValueError(f"{_MINER} requires an exact Git selector")
    if repository.lower().removesuffix(".git") != _REPOSITORY.lower().removesuffix(
        ".git"
    ):
        raise ValueError(f"Unexpected {_MINER} repository. Expected {_REPOSITORY}")

    if _COMMIT.fullmatch(selector):
        kind, value = "commit", selector.lower()
    elif _VERSION_TAG.fullmatch(selector):
        kind, value = "tag", selector
    elif selector.startswith("refs/tags/"):
        kind, value = "tag", selector.removeprefix("refs/tags/")
        if value.lower() in {"head", "latest"}:
            raise ValueError(f"{_MINER} requires an exact release tag")
    else:
        raise ValueError(
            f"{_MINER} requires a version tag, refs/tags/<tag>, or full Git commit"
        )
    return Node(
        kind="external_release",
        name=_MINER,
        payload={
            "repository": _REPOSITORY,
            "selector": {"kind": kind, "value": value},
        },
    )


def _recipe_payload(name: str) -> dict[str, object]:
    return {
        "recipe": {"name": name.replace("symbolic/", "symbolic/m2a/", 1), "version": 1},
        "purpose": "definition" if name == _ROOT else "component",
    }


def validate_symbolic_definition(
    nodes: Iterable[Node], root_id: str
) -> tuple[Node, ...]:
    """Check the M2a recipe in a graph, including exact slots and shared leaves."""
    indexed = {node.id: node for node in nodes}
    ordered = reachable_nodes((root_id,), indexed.__getitem__)
    root = ordered[-1]
    if root.kind != "composite" or root.name != _ROOT:
        raise ValueError(f"Expected M2a symbolic root {_ROOT}")

    by_name: dict[str, str] = {}
    for node in ordered:
        if by_name.setdefault(node.name, node.id) != node.id:
            raise ValueError(f"M2a requires a shared node for {node.name}")
        if node.name == _MINER and node.payload.get("repository") != _REPOSITORY:
            raise ValueError(f"Unexpected {_MINER} repository")
        if node.kind != "composite":
            continue
        required = _REQUIRED_SLOTS.get(node.name)
        if required is None or node.payload != _recipe_payload(node.name):
            raise ValueError(f"Invalid M2a recipe for {node.name}")
        actual = {
            (edge.role, edge.slot): indexed[edge.target_id].name
            for edge in node.dependencies
        }
        if actual.keys() != required.keys():
            missing = sorted(required.keys() - actual.keys())
            unexpected = sorted(actual.keys() - required.keys())
            raise ValueError(
                f"Invalid M2a slots for {node.name}. "
                f"Missing slots: {missing}. Unexpected slots: {unexpected}"
            )
        for slot, target in required.items():
            if actual[slot] != target:
                raise ValueError(
                    f"M2a slot {slot[0]}/{slot[1]} in {node.name} requires {target}"
                )
    return ordered


def build_symbolic_definition(project_root: str | Path) -> tuple[Node, ...]:
    """Build a validated definition closure from an explicit local inventory."""
    root = Path(project_root)
    release = read_hybrid_miner_release(root / "pyproject.toml")
    nodes = {release.name: release}
    sources = {
        name
        for slots in _REQUIRED_SLOTS.values()
        for (role, _), name in slots.items()
        if role == "uses-source"
    }
    for name in sorted(sources):
        data = (root / name).read_bytes()
        nodes[name] = Node(
            kind="source",
            name=name,
            payload={"sha256": sha256(data).hexdigest(), "byte_length": len(data)},
        )
    for name, slots in _REQUIRED_SLOTS.items():
        nodes[name] = Node(
            kind="composite",
            name=name,
            payload=_recipe_payload(name),
            dependencies=tuple(
                Dependency(role, slot, nodes[target].id)
                for (role, slot), target in slots.items()
            ),
        )
    return validate_symbolic_definition(nodes.values(), nodes[_ROOT].id)


def snapshot_symbolic(project_root: str | Path, registry: Registry) -> Node:
    """Atomically register the captured closure and return its definition root."""
    nodes = build_symbolic_definition(project_root)
    registry.register_batch(nodes)
    return nodes[-1]
