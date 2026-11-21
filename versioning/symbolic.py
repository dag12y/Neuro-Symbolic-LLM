"""Capture versioned symbolic definitions from local project inputs."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
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
_SELECTED_CATALOG = "ontology/selected-catalog"
_SEMANTIC_VERSION = 1
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
        ("uses-semantic", "catalog"): _SELECTED_CATALOG,
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
_CATALOG_SOURCE = {("uses-source", "catalog-loader"): "parser/ontology/catalog.py"}
_PROMPT_SOURCE = {
    ("uses-source", "prompt-builder"): "parser/semantic/catalog_prompt.py"
}
_M2B_SEMANTICS: dict[str, tuple[str, dict[tuple[str, str], str]]] = {
    _SELECTED_CATALOG: ("ontology-catalog", _CATALOG_SOURCE),
    "ontology/miner-bundle": ("miner-bundle", _CATALOG_SOURCE),
    "ontology/default-vocabulary": ("default-vocabulary", _CATALOG_SOURCE),
    "grammar/atomese": (
        "atomese-grammar",
        {
            ("uses-semantic", "vocabulary"): "ontology/default-vocabulary",
            ("uses-source", "grammar-loader"): "parser/grammar/atomese.py",
        },
    ),
    "prompt/router": ("prompt-template", _PROMPT_SOURCE),
    **{
        f"prompt/{name}": (
            "prompt-template",
            {
                **_PROMPT_SOURCE,
                ("uses-semantic", "catalog-guide"): f"ontology/prompt-guide/{name}",
            },
        )
        for name in ("general", "discourse")
    },
    **{
        f"ontology/prompt-guide/{name}": ("prompt-guide", {})
        for name in ("general", "discourse")
    },
    "prompt/student": (
        "prompt-instructions",
        {("uses-source", "prompt-builder"): "parser/student/student_prompt.py"},
    ),
    "configuration/parser-backend": (
        "parser-backend",
        {("uses-source", "profile-loader"): "parser/semantic/model_config.py"},
    ),
}
_M2B_VALUE_FIELDS = {
    "ontology-catalog": {"modules", "versions"},
    "ontology-module": {"version", "types", "relations"},
    "miner-bundle": {
        "contract_version",
        "bundle_id",
        "bundle_version",
        "ontology_versions",
        "types",
        "predicates",
    },
    "default-vocabulary": {"arities"},
    "atomese-grammar": {"fixed", "variable"},
    "prompt-template": {"prompt_version", "system_prompt"},
    "prompt-instructions": {"prompt_version", "instructions"},
}


def _semantic_recipes(
    modules: tuple[str, ...],
) -> dict[str, tuple[str, dict[tuple[str, str], str]]]:
    module_slots = {
        ("uses-semantic", f"module/{name}"): f"ontology/module/{name}"
        for name in modules
    }
    return {
        **_M2B_SEMANTICS,
        _SELECTED_CATALOG: ("ontology-catalog", {**_CATALOG_SOURCE, **module_slots}),
        **{name: ("ontology-module", {}) for name in module_slots.values()},
    }


def _is_miner_requirement(requirement: str) -> bool:
    name = _PACKAGE_NAME.match(requirement.lstrip())
    return name is not None and re.sub(r"[-_.]+", "-", name[0].lower()) == _MINER


def _miner_requirement(pyproject: Path, read_bytes: Callable[[Path], bytes]) -> str:
    config = tomllib.loads(read_bytes(pyproject).decode("utf-8"))
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


def read_hybrid_miner_release(
    pyproject: str | Path, *, read_bytes: Callable[[Path], bytes] = Path.read_bytes
) -> Node:
    """Read the declared release leaf without importing or resolving the miner."""
    requirement = _miner_requirement(Path(pyproject), read_bytes)
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


def _validate_slots(
    node: Node,
    required: Mapping[tuple[str, str], str],
    indexed: Mapping[str, Node],
    recipe: str,
) -> None:
    actual = {
        (edge.role, edge.slot): indexed[edge.target_id].name
        for edge in node.dependencies
    }
    if actual.keys() != required.keys():
        missing = sorted(required.keys() - actual.keys())
        unexpected = sorted(actual.keys() - required.keys())
        raise ValueError(
            f"Invalid {recipe} slots for {node.name}. "
            f"Missing slots: {missing}. Unexpected slots: {unexpected}"
        )
    for slot, target in required.items():
        if actual[slot] != target:
            raise ValueError(
                f"{recipe} slot {slot[0]}/{slot[1]} in {node.name} requires {target}"
            )


def _validate_m2b_semantics(
    by_name: Mapping[str, Node], indexed: Mapping[str, Node]
) -> None:
    catalog = by_name[_SELECTED_CATALOG].payload["value"]
    assert isinstance(catalog, Mapping)  # Enforced by the semantic node schema.
    modules, versions = catalog.get("modules"), catalog.get("versions")
    if (
        not isinstance(modules, tuple)
        or not modules
        or any(
            not isinstance(name, str) or not name.strip() or "/" in name
            for name in modules
        )
        or modules != tuple(sorted(set(modules)))
    ):
        raise ValueError("M2b selected catalog requires ordered, unique module names")
    if (
        not isinstance(versions, tuple)
        or len(versions) != len(modules)
        or any(
            not isinstance(version, str) or not version.strip() for version in versions
        )
        or len(set(versions)) != len(versions)
    ):
        raise ValueError("M2b selected catalog requires one unique version per module")

    expected = _semantic_recipes(modules)
    actual = {name for name, node in by_name.items() if node.kind == "semantic_content"}
    if actual != expected.keys():
        raise ValueError(
            "Invalid M2b semantic inventory. "
            f"Missing nodes: {sorted(expected.keys() - actual)}. "
            f"Unexpected nodes: {sorted(actual - expected.keys())}"
        )
    for name, (projection, slots) in expected.items():
        node = by_name[name]
        if (
            node.payload["projection"] != projection
            or node.payload["version"] != _SEMANTIC_VERSION
        ):
            raise ValueError(
                f"M2b semantic node {name} requires projection "
                f"{projection} version {_SEMANTIC_VERSION}"
            )
        _validate_slots(node, slots, indexed, "M2b")
        _validate_m2b_projection_inputs(node, projection)
    for name, version in zip(modules, versions, strict=True):
        value = by_name[f"ontology/module/{name}"].payload["value"]
        assert isinstance(value, Mapping)
        if value.get("version") != version:
            raise ValueError(
                f"M2b selected catalog version does not match module {name}"
            )


def _validate_m2b_projection_inputs(node: Node, projection: str) -> None:
    value = node.payload["value"]
    assert isinstance(value, Mapping)
    if projection == "parser-backend":
        required = {"profile", "provider", "student_role"}
        model = value.keys() & {"model", "model_env"}
        limit = value.keys() & {"max_output_tokens", "max_tokens", "max_new_tokens"}
        valid = required <= value.keys() and len(model) == len(limit) == 1
    else:
        required = _M2B_VALUE_FIELDS.get(projection, {"guide"})
        if node.name == "ontology/prompt-guide/general":
            required = {"guide", "versions"}
        valid = value.keys() == required
    if not valid:
        raise ValueError(f"Invalid M2b projection inputs for {node.name}")


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

    by_name: dict[str, Node] = {}
    for node in ordered:
        if by_name.setdefault(node.name, node).id != node.id:
            raise ValueError(f"{recipe.upper()} requires a shared node for {node.name}")
        if node.name == _MINER and node.payload.get("repository") != _REPOSITORY:
            raise ValueError(f"Unexpected {_MINER} repository")
        if node.kind != "composite":
            continue
        required = requirements.get(node.name)
        stage_recipe = "m2a" if node.name == "symbolic/mining" else recipe
        if required is None or node.payload != _recipe_payload(node.name, stage_recipe):
            raise ValueError(f"Invalid {recipe.upper()} recipe for {node.name}")
        _validate_slots(node, required, indexed, recipe.upper())
    if recipe == "m2b":
        _validate_m2b_semantics(by_name, indexed)
    return ordered


class _SnapshotInputs:
    """Hold the exact bytes used by one build and recheck them before publication."""

    def __init__(self) -> None:
        self._files: dict[Path, bytes] = {}

    def read_bytes(self, path: Path) -> bytes:
        if path not in self._files:
            self._files[path] = path.read_bytes()
        return self._files[path]

    def verify_unchanged(self) -> None:
        for path, captured in self._files.items():
            try:
                current = path.read_bytes()
            except OSError as error:
                raise ValueError(
                    f"Cannot recheck symbolic snapshot input {path}"
                ) from error
            if current != captured:
                raise ValueError(
                    f"Symbolic snapshot input changed during capture: {path}"
                )


def _source(root: Path, name: str, inputs: _SnapshotInputs) -> Node:
    data = inputs.read_bytes(root / name)
    return Node(
        kind="source",
        name=name,
        payload={"sha256": sha256(data).hexdigest(), "byte_length": len(data)},
    )


def _capture_sources(
    root: Path,
    requirements: Mapping[str, Mapping[tuple[str, str], str]],
    inputs: _SnapshotInputs,
) -> dict[str, Node]:
    release = read_hybrid_miner_release(
        root / "pyproject.toml", read_bytes=inputs.read_bytes
    )
    nodes = {release.name: release}
    sources = {
        name
        for slots in requirements.values()
        for (role, _), name in slots.items()
        if role == "uses-source"
    }
    for name in sorted(sources):
        nodes[name] = _source(root, name, inputs)
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
    inputs = _SnapshotInputs()
    nodes = _capture_sources(Path(project_root), _REQUIRED_SLOTS, inputs)
    definition = _compose(nodes, _REQUIRED_SLOTS, "m2a")
    inputs.verify_unchanged()
    return definition


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
    inputs = _SnapshotInputs()
    nodes = _capture_sources(root, _M2B_SLOTS, inputs)
    ontology = root / "parser/ontology"
    config = root / "configs/parser_config"
    module_paths = sorted(ontology.glob("*.yaml"))
    catalog = load_catalog_inputs(
        module_paths,
        ontology / "bundles/semantic-extraction-v1.yaml",
        root / ontology_bundle if ontology_bundle is not None else None,
        read_bytes=inputs.read_bytes,
    )
    recipes = _semantic_recipes(catalog.selection)

    def content(name: str, value: dict[str, object]) -> None:
        projection, slots = recipes[name]
        nodes[name] = Node(
            kind="semantic_content",
            name=name,
            payload={
                "projection": projection,
                "version": _SEMANTIC_VERSION,
                "value": value,
            },
            dependencies=tuple(
                Dependency(role, slot, nodes[target].id)
                for (role, slot), target in slots.items()
            ),
        )

    sources = {
        target
        for _, slots in recipes.values()
        for (role, _), target in slots.items()
        if role == "uses-source"
    }
    for path in sorted(sources - nodes.keys()):
        nodes[path] = _source(root, path, inputs)

    for name in catalog.selection:
        content(f"ontology/module/{name}", catalog.modules[name])
    content(
        _SELECTED_CATALOG,
        {
            "modules": list(catalog.selection),
            "versions": list(catalog.selected.versions),
        },
    )
    content(
        "ontology/miner-bundle",
        miner_bundle_projection(catalog.miner_catalog, catalog.miner_manifest),
    )
    vocabulary = {
        name: relation.arity for name, relation in catalog.full.relations.items()
    }
    content("ontology/default-vocabulary", {"arities": vocabulary})
    content(
        "grammar/atomese",
        grammar_projection(
            config / "atomese_grammar.yaml", vocabulary, read_bytes=inputs.read_bytes
        ),
    )
    for name, guide in prompt_guides(catalog.selected).items():
        content(f"ontology/prompt-guide/{name}", guide)
    for name, template in prompt_templates(
        config, read_bytes=inputs.read_bytes
    ).items():
        content(f"prompt/{name}", template)
    content("prompt/student", student_instructions(catalog.selected))
    content(
        "configuration/parser-backend",
        backend_projection(config / "model_backend.yaml", read_bytes=inputs.read_bytes),
    )
    definition = _compose(nodes, _M2B_SLOTS, "m2b")
    inputs.verify_unchanged()
    if sorted(ontology.glob("*.yaml")) != module_paths:
        raise ValueError("Symbolic snapshot ontology module set changed during capture")
    return definition


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
