"""Loader-aligned projections for the symbolic parser definition."""

from __future__ import annotations

import math
import ntpath
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from struct import pack
from tempfile import TemporaryDirectory
from typing import Any

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode
from yaml.nodes import Node as YAMLNode

from parser.ontology.catalog import (
    BundleManifest,
    OntologyCatalog,
    load_manifest_catalog,
    load_ontology_catalog,
)
from parser.semantic.catalog_prompt import (
    _catalog_guide,
    _discourse_catalog_guide,
    _general_catalog_versions,
    _require_placeholders,
)
from parser.student.student_prompt import STUDENT_PROMPT, _instructions

from .hashing import canonical_json
from .nodes import Node, reachable_nodes

_YAML_TAGS = {
    f"tag:yaml.org,2002:{tag}"
    for tag in ("map", "seq", "str", "int", "float", "bool", "null")
}


class _ProjectionLoader(yaml.SafeLoader):
    """Use SafeLoader scalars while rejecting ambiguous structure."""

    def compose_node(self, parent: YAMLNode | None, index: Any) -> YAMLNode:
        event = self.peek_event()
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ValueError("YAML anchors and aliases are unsupported")
        node = super().compose_node(parent, index)
        if node is None:
            raise ValueError("Expected a YAML node")
        if node.tag not in _YAML_TAGS:
            raise ValueError(f"Unsupported YAML tag: {node.tag}")
        return node

    def construct_mapping(self, node: YAMLNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ValueError("Expected a YAML mapping node")
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in keys:
                raise ValueError(f"Duplicate YAML key: {key!r}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_yaml(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = yaml.load(text, Loader=_ProjectionLoader)
        return text, _mapping(value, path.name)
    except (yaml.YAMLError, ValueError, RecursionError) as error:
        raise ValueError(f"{path.name}: {error}") from error


def _binary64(value: int | float) -> str:
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("Expected a finite binary64 value") from error
    if not math.isfinite(number):
        raise ValueError("Expected a finite binary64 value")
    return pack(">d", number).hex()


@dataclass(frozen=True)
class CatalogInputs:
    modules: dict[str, dict[str, object]]
    full: OntologyCatalog
    selected: OntologyCatalog
    selection: tuple[str, ...]
    miner_catalog: OntologyCatalog
    miner_manifest: BundleManifest


def load_catalog_inputs(
    directory: Path, default_manifest: Path, selected_manifest: Path | None
) -> CatalogInputs:
    """Validate exactly the captured text using the runtime catalog loaders."""
    documents = {}
    with TemporaryDirectory(prefix="nsllm-catalog-") as temporary:
        captured = Path(temporary)
        modules = captured / "modules"
        modules.mkdir()
        for path in sorted(directory.glob("*.yaml")):
            text, documents[path.stem] = _read_yaml(path)
            (modules / path.name).write_text(text, encoding="utf-8")
        full = load_ontology_catalog(modules)

        default_copy = captured / "default.yaml"
        default_copy.write_text(_read_yaml(default_manifest)[0], encoding="utf-8")
        miner_catalog, miner_manifest = load_manifest_catalog(
            default_copy, ontology_directory=modules
        )
        selected, manifest = miner_catalog, miner_manifest
        if selected_manifest is not None and selected_manifest != default_manifest:
            selected_copy = captured / "selected.yaml"
            selected_copy.write_text(_read_yaml(selected_manifest)[0], encoding="utf-8")
            selected, manifest = load_manifest_catalog(
                selected_copy, ontology_directory=modules
            )

    projections: dict[str, dict[str, object]] = {
        name: {
            "version": document["version"],
            "types": sorted(value.strip() for value in document.get("types", [])),
            "relations": {},
        }
        for name, document in documents.items()
    }
    relations: dict[str, dict[str, object]] = {name: {} for name in documents}
    for relation in full.relations.values():
        relations[relation.ontology][relation.name] = {
            "arity": relation.arity,
            "arguments": [
                {"name": argument.name, "allowed_types": list(argument.allowed_types)}
                for argument in relation.arguments
            ],
            "confidence_range_binary64": [
                _binary64(bound) for bound in relation.confidence_range
            ],
            "extraction_instruction": relation.extraction_instruction,
        }
    for name, projection in projections.items():
        projection["relations"] = relations[name]
    return CatalogInputs(
        projections,
        full,
        selected,
        tuple(sorted(manifest.modules)),
        miner_catalog,
        miner_manifest,
    )


def miner_bundle_projection(
    catalog: OntologyCatalog, manifest: BundleManifest
) -> dict[str, object]:
    """Capture the inputs to OntologyBundle.model_validate without importing it."""
    relations = sorted(catalog.relations.values(), key=lambda relation: relation.name)
    return {
        "contract_version": "ontology-bundle-v1",
        "bundle_id": manifest.bundle_id,
        "bundle_version": manifest.bundle_version,
        "ontology_versions": list(catalog.versions),
        "types": sorted(catalog.types | {"String"}),
        "predicates": [
            {
                "name": relation.name,
                "ontology_version": relation.ontology_version,
                "argument_types": [
                    list(arg.allowed_types) for arg in relation.arguments
                ],
            }
            for relation in relations
        ],
    }


def grammar_projection(path: Path, vocabulary: dict[str, int]) -> dict[str, object]:
    """Project the arity tables read by Atomese validation."""
    _, document = _read_yaml(path)
    structural = _mapping(document.get("structural"), "Structural grammar")
    operators = _mapping(document.get("operators"), "Grammar operators")
    if (structural.keys() & operators.keys()) or (
        (structural.keys() | operators.keys()) & vocabulary.keys()
    ):
        raise ValueError("Atomese predicates must be unique across grammar and catalog")
    fixed: dict[str, int] = {}
    variable: dict[str, list[int | None]] = {}
    for name, value in {**structural, **operators}.items():
        definition = _mapping(value, f"Atomese predicate {name!r}")
        arity = definition.get("arity")
        if definition.get("variable_arity") is True:
            arity = None
        if arity is not None:
            if not isinstance(arity, int) or isinstance(arity, bool) or arity < 1:
                raise ValueError(f"Invalid Atomese arity for {name!r}")
            fixed[name] = arity
            continue
        lower, upper = definition.get("min_arity"), definition.get("max_arity")
        if (
            not isinstance(lower, int)
            or isinstance(lower, bool)
            or lower < 0
            or (
                upper is not None
                and (
                    not isinstance(upper, int)
                    or isinstance(upper, bool)
                    or upper < lower
                )
            )
        ):
            raise ValueError(f"Invalid variable Atomese arity for {name!r}")
        variable[name] = [lower, upper]
    return {"fixed": fixed, "variable": variable}


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def prompt_templates(directory: Path) -> dict[str, dict[str, object]]:
    """Capture loaded template strings and labels used by the routed parser."""
    configs = {
        "router": ("parser_prompt.yaml", ("{source_text}",)),
        "general": (
            "general_prompt.yaml",
            ("{ontology_versions}", "{catalog_guide}", "{context}", "{source_text}"),
        ),
        "discourse": ("discourse_prompt.yaml", ("{discourse_catalog_guide}",)),
    }
    prompts: dict[str, dict[str, object]] = {}
    for name, (filename, placeholders) in configs.items():
        _, config = _read_yaml(directory / filename)
        version = _text(config.get("prompt_version"), f"{name} prompt version")
        template = _text(config.get("system_prompt"), f"{name} prompt text")
        _require_placeholders(template, placeholders, name)
        prompts[name] = {"prompt_version": version, "system_prompt": template}
    return prompts


def prompt_guides(catalog: OntologyCatalog) -> dict[str, dict[str, object]]:
    """Use the runtime's exact ontology text and version ordering."""
    has_discourse = any(
        relation.ontology == "discourse" for relation in catalog.relations.values()
    )
    return {
        "general": {
            "versions": list(_general_catalog_versions(catalog)),
            "guide": _catalog_guide(catalog),
        },
        "discourse": {
            "guide": (_discourse_catalog_guide(catalog) if has_discourse else None),
        },
    }


def student_instructions(catalog: OntologyCatalog) -> dict[str, object]:
    return {"prompt_version": STUDENT_PROMPT, "instructions": _instructions(catalog)}


_PROVIDERS = {
    # Required env references, optional env references, token setting and default.
    "openai": (("api_key_env",), (), "max_output_tokens", 256),
    "anthropic": (("api_key_env",), (), "max_tokens", 256),
    "gemini": (("api_key_env",), (), "max_output_tokens", 256),
    "ollama": ((), ("host_env",), "max_output_tokens", 256),
    "openai-compatible": (
        ("base_url_env",),
        ("api_key_env",),
        "max_output_tokens",
        256,
    ),
    "transformers-local": ((), ("hf_token_env",), "max_new_tokens", 256),
}


def backend_projection(path: Path) -> dict[str, object]:
    """Project the active reference backend without reading the environment."""
    _, document = _read_yaml(path)
    active = _text(document.get("active"), "Active profile")
    profiles = _mapping(document.get("profiles"), "Model profiles")
    profile = _mapping(profiles.get(active), f"Profile {active!r}")
    provider = _text(profile.get("provider"), "Provider").strip()
    if provider not in _PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider!r}")
    model, model_env = profile.get("model"), profile.get("model_env")
    value: dict[str, object] = {"profile": active, "provider": provider}
    if model_env is None:
        locator = _text(model, "Model").strip()
        if (
            ntpath.isabs(locator)
            or ntpath.splitdrive(locator)[0]
            or locator.startswith("~")
        ):
            raise ValueError("Model identity requires a portable model locator")
        value["model"] = locator
    elif model is None:
        value["model_env"] = _text(model_env, "Model environment variable").strip()
    else:
        raise ValueError("Model profile must select either model or model_env")
    required, optional, token_key, default = _PROVIDERS[provider]
    for key in (*required, *optional):
        setting = profile.get(key)
        if setting is not None or key in required:
            value[key] = _text(setting, key).strip()
    limit = profile.get(token_key, default)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f"{token_key} must be a positive integer")
    value[token_key] = limit
    value["student_role"] = profile.get("role") == "student"

    if provider == "ollama":
        for key, expected in (("think", bool), ("seed", int), ("num_ctx", int)):
            setting = profile.get(key)
            if setting is not None and type(setting) is not expected:
                raise ValueError(f"{key} must be {expected.__name__} or null")
            value[key] = setting
        temperature = profile.get("temperature")
        if temperature is not None:
            if type(temperature) not in (int, float):
                raise ValueError("temperature must be a number")
            value["temperature_binary64"] = _binary64(temperature)
    if provider == "transformers-local":
        add_tokens = profile.get("add_special_tokens")
        if add_tokens is not None and not isinstance(add_tokens, bool):
            raise ValueError("add_special_tokens must be boolean or null")
        value["add_special_tokens"] = True if add_tokens is None else add_tokens
    return value


@dataclass(frozen=True)
class SemanticChange:
    category: str
    name: str
    change: str


def _parser_inventory(nodes: Iterable[Node]) -> dict[tuple[str, str], object]:
    indexed = {node.id: node for node in nodes}
    parsers = [node.id for node in indexed.values() if node.name == "symbolic/parser"]
    if len(parsers) != 1:
        raise ValueError("Expected a graph with one parser definition")
    entries: dict[tuple[str, str], object] = {}
    for node in reachable_nodes(parsers, indexed.__getitem__):
        if node.kind == "source":
            entries["implementation", node.name] = node.id
        elif node.kind == "semantic_content":
            payload = _mapping(node.to_dict()["payload"], "Semantic payload")
            projection, value = payload["projection"], payload["value"]
            if projection == "ontology-module":
                module = node.name.removeprefix("ontology/module/")
                entries["ontology-module", module] = {
                    "types": value["types"],
                    "version": value["version"],
                }
                for name, relation in value["relations"].items():
                    entries["ontology-entry", f"{module}/{name}"] = relation
            else:
                entries[projection, node.name] = value
    return entries


def compare_parser_definitions(
    before: Iterable[Node], after: Iterable[Node]
) -> tuple[SemanticChange, ...]:
    """Report changed projections and source identities of two parser closures."""
    old, new = _parser_inventory(before), _parser_inventory(after)
    changes = []
    for key in sorted(old.keys() | new.keys()):
        category, name = key
        if key not in old:
            change = "added"
        elif key not in new:
            change = "removed"
        elif canonical_json(old[key]) != canonical_json(new[key]):
            change = "modified"
        else:
            continue
        changes.append(SemanticChange(category, name, change))
    return tuple(changes)
