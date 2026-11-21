"""Capture versioned symbolic definitions from local project inputs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import tomllib

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
_PARSER_SUPPORT = {
    ("uses-source", name): f"parser/semantic/{name}.py"
    for name in ("backends", "metta_renderer", "normalization", "output_cleanup")
}
_M2B_SLOTS = {
    **_REQUIRED_SLOTS,
    "symbolic/parser": {
        **_REQUIRED_SLOTS["symbolic/parser"],
        **_PARSER_SUPPORT,
        ("uses-source", "student-validation"): "parser/semantic/catalog_student.py",
        ("uses-semantic", "catalog"): "ontology/selected-catalog",
        ("uses-semantic", "grammar"): "grammar/atomese",
        ("uses-semantic", "backend"): "configuration/parser-backend",
        **{
            ("uses-semantic", f"prompt/{name}"): f"prompt/{name}"
            for name in ("router", "general", "discourse", "student")
        },
    },
    "symbolic/adapter": {
        **_REQUIRED_SLOTS["symbolic/adapter"],
        ("uses-semantic", "ontology-bundle"): "ontology/miner-bundle",
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


def _recipe_payload(name: str, recipe: str = "m2a") -> dict[str, object]:
    identifier = name.replace("symbolic/", f"symbolic/{recipe}/", 1)
    return {
        "recipe": {"name": identifier, "version": 1},
        "purpose": "definition" if name == _ROOT else "component",
    }


def validate_symbolic_definition(
    nodes: Iterable[Node], root_id: str
) -> tuple[Node, ...]:
    """Check definition recipe slots, target identities, and shared leaves."""
    indexed = {node.id: node for node in nodes}
    ordered = reachable_nodes((root_id,), indexed.__getitem__)
    root = ordered[-1]
    if root.kind != "composite" or root.name != _ROOT:
        raise ValueError(f"Expected symbolic root {_ROOT}")
    recipe = "m2b" if root.payload == _recipe_payload(_ROOT, "m2b") else "m2a"
    requirements = _M2B_SLOTS if recipe == "m2b" else _REQUIRED_SLOTS

    by_name: dict[str, str] = {}
    for node in ordered:
        if by_name.setdefault(node.name, node.id) != node.id:
            raise ValueError(f"{recipe.upper()} requires a shared node for {node.name}")
        if node.name == _MINER and node.payload.get("repository") != _REPOSITORY:
            raise ValueError(f"Unexpected {_MINER} repository")
        if node.kind != "composite":
            continue
        required = requirements.get(node.name)
        stage_recipe = "m2a" if node.name == "symbolic/mining" else recipe
        if required is None or node.payload != _recipe_payload(node.name, stage_recipe):
            raise ValueError(f"Invalid {recipe.upper()} recipe for {node.name}")
        actual = {
            (edge.role, edge.slot): indexed[edge.target_id].name
            for edge in node.dependencies
        }
        if actual.keys() != required.keys():
            missing = sorted(required.keys() - actual.keys())
            unexpected = sorted(actual.keys() - required.keys())
            raise ValueError(
                f"Invalid {recipe.upper()} slots for {node.name}. "
                f"Missing slots: {missing}. Unexpected slots: {unexpected}"
            )
        for slot, target in required.items():
            if actual[slot] != target:
                raise ValueError(
                    f"{recipe.upper()} slot {slot[0]}/{slot[1]} in {node.name} "
                    f"requires {target}"
                )
    return ordered


def _source(root: Path, name: str) -> Node:
    data = (root / name).read_bytes()
    return Node(
        kind="source",
        name=name,
        payload={"sha256": sha256(data).hexdigest(), "byte_length": len(data)},
    )


def _capture_sources(
    root: Path, requirements: Mapping[str, Mapping[tuple[str, str], str]]
) -> dict[str, Node]:
    release = read_hybrid_miner_release(root / "pyproject.toml")
    nodes = {release.name: release}
    sources = {
        name
        for slots in requirements.values()
        for (role, _), name in slots.items()
        if role == "uses-source"
    }
    for name in sorted(sources):
        nodes[name] = _source(root, name)
    return nodes


def _compose(
    nodes: dict[str, Node],
    requirements: Mapping[str, Mapping[tuple[str, str], str]],
    recipe: str,
) -> tuple[Node, ...]:
    for name, slots in requirements.items():
        stage_recipe = "m2a" if name == "symbolic/mining" else recipe
        nodes[name] = Node(
            kind="composite",
            name=name,
            payload=_recipe_payload(name, stage_recipe),
            dependencies=tuple(
                Dependency(role, slot, nodes[target].id)
                for (role, slot), target in slots.items()
            ),
        )
    return validate_symbolic_definition(nodes.values(), nodes[_ROOT].id)


def build_m2a_symbolic_definition(project_root: str | Path) -> tuple[Node, ...]:
    """Reconstruct the historical source-only recipe without changing its IDs."""
    nodes = _capture_sources(Path(project_root), _REQUIRED_SLOTS)
    return _compose(nodes, _REQUIRED_SLOTS, "m2a")


def build_symbolic_definition(
    project_root: str | Path, *, ontology_bundle: str | Path | None = None
) -> tuple[Node, ...]:
    """Build the M2b semantic definition for the selected parser bundle."""
    from .semantic import (
        backend_projection,
        grammar_projection,
        load_catalog_inputs,
        miner_bundle_projection,
        prompt_guides,
        prompt_templates,
        student_instructions,
    )

    root = Path(project_root)
    nodes = _capture_sources(root, _M2B_SLOTS)
    ontology = root / "parser/ontology"
    config = root / "configs/parser_config"
    catalog = load_catalog_inputs(
        ontology,
        ontology / "bundles/semantic-extraction-v1.yaml",
        root / ontology_bundle if ontology_bundle is not None else None,
    )

    def content(
        name: str, projection: str, value: dict[str, object], *edges: Dependency
    ) -> None:
        nodes[name] = Node(
            kind="semantic_content",
            name=name,
            payload={"projection": projection, "version": 1, "value": value},
            dependencies=edges,
        )

    for path in (
        "parser/ontology/catalog.py",
        "parser/grammar/atomese.py",
        "parser/semantic/catalog_prompt.py",
        "parser/semantic/model_config.py",
        "parser/student/student_prompt.py",
    ):
        nodes[path] = _source(root, path)

    def edge(role: str, slot: str, target: str) -> Dependency:
        return Dependency(role, slot, nodes[target].id)

    loader = edge("uses-source", "catalog-loader", "parser/ontology/catalog.py")
    for name in catalog.selection:
        content(f"ontology/module/{name}", "ontology-module", catalog.modules[name])
    content(
        "ontology/selected-catalog",
        "ontology-catalog",
        {
            "modules": list(catalog.selection),
            "versions": list(catalog.selected.versions),
        },
        loader,
        *(
            edge("uses-semantic", f"module/{name}", f"ontology/module/{name}")
            for name in catalog.selection
        ),
    )
    content(
        "ontology/miner-bundle",
        "miner-bundle",
        miner_bundle_projection(catalog.miner_catalog, catalog.miner_manifest),
        loader,
    )
    vocabulary = {
        name: relation.arity for name, relation in catalog.full.relations.items()
    }
    content(
        "ontology/default-vocabulary",
        "default-vocabulary",
        {"arities": vocabulary},
        loader,
    )
    content(
        "grammar/atomese",
        "atomese-grammar",
        grammar_projection(config / "atomese_grammar.yaml", vocabulary),
        edge("uses-semantic", "vocabulary", "ontology/default-vocabulary"),
        edge("uses-source", "grammar-loader", "parser/grammar/atomese.py"),
    )
    for name, guide in prompt_guides(catalog.selected).items():
        content(f"ontology/prompt-guide/{name}", "prompt-guide", guide)
    prompt_builder = edge(
        "uses-source", "prompt-builder", "parser/semantic/catalog_prompt.py"
    )
    for name, template in prompt_templates(config).items():
        guide_name = f"ontology/prompt-guide/{name}"
        edges = [prompt_builder]
        if guide_name in nodes:
            edges.append(edge("uses-semantic", "catalog-guide", guide_name))
        content(f"prompt/{name}", "prompt-template", template, *edges)
    content(
        "prompt/student",
        "prompt-instructions",
        student_instructions(catalog.selected),
        edge("uses-source", "prompt-builder", "parser/student/student_prompt.py"),
    )
    content(
        "configuration/parser-backend",
        "parser-backend",
        backend_projection(config / "model_backend.yaml"),
        edge("uses-source", "profile-loader", "parser/semantic/model_config.py"),
    )
    return _compose(nodes, _M2B_SLOTS, "m2b")


def snapshot_symbolic(
    project_root: str | Path,
    registry: Registry,
    *,
    recipe: str = "m2b",
    ontology_bundle: str | Path | None = None,
) -> Node:
    """Atomically register the captured closure and return its definition root."""
    if recipe == "m2a" and ontology_bundle is None:
        nodes = build_m2a_symbolic_definition(project_root)
    elif recipe == "m2b":
        nodes = build_symbolic_definition(project_root, ontology_bundle=ontology_bundle)
    else:
        raise ValueError("Expected M2b or M2a without an ontology bundle override")
    registry.register_batch(nodes)
    return nodes[-1]
