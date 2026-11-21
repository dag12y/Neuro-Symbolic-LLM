from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from typing import Any, cast

import pytest
import yaml

from parser.grammar.atomese import ARITY, VARIABLE_ARITY
from parser.ontology.catalog import load_manifest_catalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from versioning.hashing import canonical_json, hash_file
from versioning.manifest import MANIFEST_VERSION, export_manifest, verify_manifest
from versioning.nodes import Dependency, Node, reachable_nodes
from versioning.registry import Registry
from versioning.semantic import (
    SemanticChange,
    backend_projection,
    compare_parser_definitions,
)
from versioning.symbolic import (
    build_m2a_symbolic_definition,
    build_symbolic_definition,
    snapshot_symbolic,
    validate_symbolic_definition,
)

_REPO = Path(__file__).resolve().parents[2]
_ONTOLOGY = Path("parser/ontology")
_CONFIG = Path("configs/parser_config")
_BUNDLE = _ONTOLOGY / "bundles/semantic-extraction-v1.yaml"
_MODULES = {"discourse", "general", "lexical", "narrative", "program", "proof", "upper"}
_PARSER = "symbolic/parser"
_ADAPTER = "symbolic/adapter"
_MINING = "symbolic/mining"
_ROOT = "symbolic/definition"
_GENERAL = _ONTOLOGY / "general.yaml"


@pytest.fixture(scope="module")
def source_paths() -> set[Path]:
    return {
        Path(node.name)
        for node in build_symbolic_definition(_REPO)
        if node.kind == "source"
    }


@pytest.fixture
def project(tmp_path: Path, source_paths: set[Path]) -> Path:
    root = tmp_path / "project"
    configs = {
        _CONFIG / name
        for name in (
            "parser_prompt.yaml",
            "general_prompt.yaml",
            "discourse_prompt.yaml",
            "model_backend.yaml",
            "atomese_grammar.yaml",
        )
    }
    ontologies = {_ONTOLOGY / f"{name}.yaml" for name in _MODULES}
    for name in source_paths | configs | ontologies | {Path("pyproject.toml"), _BUNDLE}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(_REPO / name, path)
    return root


def _by_name(nodes: tuple[Node, ...]) -> dict[str, Node]:
    return {node.name: node for node in nodes}


def _changed(before: tuple[Node, ...], after: tuple[Node, ...]) -> set[str]:
    old, new = _by_name(before), _by_name(after)
    return {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _write(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_default_inventory_matches_runtime_grammar_and_prompt_guide() -> None:
    """The default closure covers seven modules and the actual runtime projections."""
    nodes = _by_name(build_symbolic_definition(_REPO))
    assert {
        node.name.removeprefix("ontology/module/")
        for node in nodes.values()
        if node.payload.get("projection") == "ontology-module"
    } == _MODULES
    grammar = cast(dict[str, Any], nodes["grammar/atomese"].payload["value"])
    vocabulary = cast(
        dict[str, Any], nodes["ontology/default-vocabulary"].payload["value"]
    )
    assert {**grammar["fixed"], **vocabulary["arities"]} == ARITY
    assert grammar["variable"] == VARIABLE_ARITY
    catalog, _ = load_manifest_catalog()
    prompt = build_catalog_document_prompt("example", catalog=catalog)
    guide = cast(
        dict[str, str], nodes["ontology/prompt-guide/general"].payload["value"]
    )
    assert guide["guide"] in prompt.text


@pytest.mark.parametrize(
    "change", ["comment", "formatting", "mapping-order", "type-set"]
)
def test_equivalent_yaml_keeps_ids(project: Path, change: str) -> None:
    """Comments, YAML spelling, mapping order, and set ordering preserve values."""
    before = build_symbolic_definition(project)
    path = project / _GENERAL
    old_raw = hash_file(path)
    if change == "comment":
        for path_to_edit in [
            path,
            project / _BUNDLE,
            project / _CONFIG / "general_prompt.yaml",
            project / _CONFIG / "atomese_grammar.yaml",
            project / _CONFIG / "model_backend.yaml",
        ]:
            original = path_to_edit.read_text(encoding="utf-8")
            path_to_edit.write_text("# tracker comment\n" + original, encoding="utf-8")
    elif change == "type-set":
        path = project / _ONTOLOGY / "upper.yaml"
        old_raw = hash_file(path)
        document = _load(path)
        document["types"].reverse()
        _write(path, document)
    else:
        document = _load(path)
        if change == "mapping-order":
            document["relations"] = dict(reversed(list(document["relations"].items())))
            document = dict(reversed(list(document.items())))
        rendered = yaml.safe_dump(
            document, sort_keys=False, default_flow_style=change == "formatting"
        )
        path.write_text(rendered, encoding="utf-8")
    assert hash_file(path) != old_raw
    after = build_symbolic_definition(project)
    assert _changed(before, after) == set()
    assert compare_parser_definitions(before, after) == ()


@pytest.mark.parametrize(
    ("change", "expected_extra"),
    [
        ("argument-order", {"ontology/miner-bundle", _ADAPTER}),
        ("type-union", {"ontology/miner-bundle", _ADAPTER}),
        ("confidence", set()),
        ("instruction", set()),
        (
            "add",
            {
                "ontology/miner-bundle",
                _ADAPTER,
                "ontology/default-vocabulary",
                "grammar/atomese",
            },
        ),
        (
            "remove",
            {
                "ontology/miner-bundle",
                _ADAPTER,
                "ontology/default-vocabulary",
                "grammar/atomese",
            },
        ),
    ],
)
def test_module_change_propagates_to_its_consumers(
    project: Path, change: str, expected_extra: set[str]
) -> None:
    """Prompt wording is narrower than the bundle, and only arity affects vocabulary."""
    before = build_symbolic_definition(project)
    path = project / _GENERAL
    doc = _load(path)
    relations = doc["relations"]
    item = relations["Inheritance"]
    entry, status = "Inheritance", "modified"
    if change == "argument-order":
        entry = "UsedFor"
        relations[entry]["arguments"].reverse()
    elif change == "type-union":
        # Reverse an existing ordered union. The labels are structurally significant.
        entry, item = next(
            (name, relation)
            for name, relation in relations.items()
            if any("|" in arg["type"] for arg in relation["arguments"])
        )
        arg = next(arg for arg in item["arguments"] if "|" in arg["type"])
        arg["type"] = "|".join(reversed(arg["type"].split("|")))
    elif change == "confidence":
        item["confidence_range"][0] = 0.125
    elif change == "instruction":
        item["extraction_instruction"] += " Use the explicit relationship."
    elif change == "add":
        entry, status = "NewPredicate", "added"
        relations[entry] = deepcopy(item)
    else:
        entry, status = "MemberOf", "removed"
        relations.pop(entry)
    _write(path, doc)
    after = build_symbolic_definition(project)
    assert _changed(before, after) == {
        "ontology/module/general",
        "ontology/selected-catalog",
        "ontology/prompt-guide/general",
        "prompt/general",
        "prompt/student",
        _PARSER,
        _ROOT,
        *expected_extra,
    }
    assert SemanticChange("ontology-entry", f"general/{entry}", status) in (
        compare_parser_definitions(before, after)
    )


def test_prompt_scalar_hash_character_is_content(project: Path) -> None:
    """A hash inside the template is passed to the parser, unlike a YAML comment."""
    before = build_symbolic_definition(project)
    path = project / _CONFIG / "parser_prompt.yaml"
    template = _load(path)["system_prompt"]
    text = path.read_text(encoding="utf-8")
    path.write_text(text.rstrip() + "\n  # literal instruction\n", encoding="utf-8")
    assert _load(path)["system_prompt"] != template
    assert "# literal instruction" in _load(path)["system_prompt"]
    after = build_symbolic_definition(project)
    assert _changed(before, after) == {"prompt/router", _PARSER, _ROOT}
    assert SemanticChange("prompt-template", "prompt/router", "modified") in (
        compare_parser_definitions(before, after)
    )


def test_confidence_preserves_signed_zero_in_prompt_guidance(project: Path) -> None:
    """The runtime formats a negative zero differently even though it compares equal."""
    before = build_symbolic_definition(project)
    path = project / _GENERAL
    document = _load(path)
    document["relations"]["Inheritance"]["confidence_range"][0] = -0.0
    _write(path, document)
    after = build_symbolic_definition(project)
    module = _by_name(after)["ontology/module/general"]
    value = cast(dict[str, Any], module.payload["value"])
    assert value["relations"]["Inheritance"]["confidence_range_binary64"][0] == (
        "8000000000000000"
    )
    assert _changed(before, after) == {
        "ontology/module/general",
        "ontology/selected-catalog",
        "ontology/prompt-guide/general",
        "prompt/general",
        "prompt/student",
        _PARSER,
        _ROOT,
    }


@pytest.mark.parametrize(
    ("extra", "error"),
    [
        ('version: "again"\n', "Duplicate YAML key"),
        ("extra: &value [1]\n", "anchors and aliases"),
        ("extra: *other\n", "anchors and aliases"),
        ("<<: {extra: 1}\n", "Unsupported YAML tag"),
        ("extra: 2026-11-21\n", "Unsupported YAML tag"),
        ("false: text\n", "YAML mapping keys must be strings"),
    ],
)
def test_rejects_ambiguous_or_unsupported_yaml(
    project: Path, extra: str, error: str
) -> None:
    """Ambiguous documents do not silently acquire a semantic identity."""
    path = project / _GENERAL
    path.write_text(path.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        build_symbolic_definition(project)


def test_selected_catalog_and_full_grammar_have_distinct_inputs(project: Path) -> None:
    """Selection uses sorted filenames, while grammar still sees unselected arities."""
    default = build_symbolic_definition(project)
    manifest = _load(project / _BUNDLE)
    manifest["modules"].pop("lexical")
    alternative = project / "alternative.yaml"
    _write(alternative, manifest)
    selected = build_symbolic_definition(project, ontology_bundle="alternative.yaml")
    assert _changed(default, selected) == {
        "ontology/module/lexical",
        "ontology/selected-catalog",
        "ontology/prompt-guide/general",
        "prompt/general",
        "prompt/student",
        _PARSER,
        _ROOT,
    }
    manifest["modules"] = dict(reversed(list(manifest["modules"].items())))
    _write(alternative, manifest)
    assert build_symbolic_definition(project, ontology_bundle=alternative) == selected

    path = project / _ONTOLOGY / "lexical.yaml"
    lexical = _load(path)
    lexical["relations"]["Kick"]["arity"] = 2
    lexical["relations"]["Kick"]["arguments"].pop()
    _write(path, lexical)
    changed = build_symbolic_definition(project, ontology_bundle=alternative)
    assert _changed(selected, changed) == {
        "ontology/default-vocabulary",
        "grammar/atomese",
        "ontology/miner-bundle",
        _ADAPTER,
        _PARSER,
        _ROOT,
    }


def test_grammar_only_tracks_effective_arity(project: Path) -> None:
    """Roles and descriptions are metadata, but validator arity is meaningful."""
    before = build_symbolic_definition(project)
    path = project / _CONFIG / "atomese_grammar.yaml"
    grammar = _load(path)
    grammar["operators"]["Equal"]["roles"].reverse()
    grammar["operators"]["Equal"]["description"] = "different annotation"
    _write(path, grammar)
    assert build_symbolic_definition(project) == before
    grammar["structural"]["List"]["max_arity"] += 1
    _write(path, grammar)
    assert _changed(before, build_symbolic_definition(project)) == {
        "grammar/atomese",
        _PARSER,
        _ROOT,
    }


def test_backend_profile_tracks_effective_declaration(project: Path) -> None:
    """The active provider matters, while inactive settings and dotenv values do not."""
    before = build_symbolic_definition(project)
    path = project / _CONFIG / "model_backend.yaml"
    config = _load(path)
    config["profiles"]["ollama"]["num_ctx"] += 1
    _write(path, config)
    assert build_symbolic_definition(project) == before
    config["profiles"][config["active"]]["temperature"] = 0.125
    _write(path, config)
    assert _changed(before, build_symbolic_definition(project)) == {
        "configuration/parser-backend",
        _PARSER,
        _ROOT,
    }


@pytest.mark.parametrize(
    ("file", "nodes"),
    [
        ("parser/semantic/schema.py", {_PARSER, _ADAPTER}),
        ("parser/semantic/semantic_parser.py", {_PARSER}),
    ],
)
def test_implementation_bytes_and_shared_schema(
    project: Path, file: str, nodes: set[str]
) -> None:
    """Python files use byte identity, and the adapter uses the shared contract."""
    old = build_symbolic_definition(project)
    with (project / file).open("ab") as stream:
        stream.write(b"\n# implementation byte change\n")
    new = build_symbolic_definition(project)
    assert _changed(old, new) == {file, _ROOT, *nodes}
    assert SemanticChange("implementation", file, "modified") in (
        compare_parser_definitions(old, new)
    )


def test_release_update_is_independent_of_parser(project: Path) -> None:
    """Only adoption of a new miner pin affects the external-release consumers."""
    old = build_symbolic_definition(project)
    path = project / "pyproject.toml"
    content = path.read_text(encoding="utf-8")
    path.write_text(content + "\n[tool.fixture]\nother = true\n", encoding="utf-8")
    assert build_symbolic_definition(project) == old
    path.write_text(content.replace(".git@v0.1.1", ".git@v0.1.0"), encoding="utf-8")
    assert _changed(old, build_symbolic_definition(project)) == {
        "hybrid-miner",
        _ADAPTER,
        _MINING,
        _ROOT,
    }


def test_manifest_history_offline_and_idempotent_registration(
    project: Path, tmp_path: Path
) -> None:
    """Old and new closures coexist, and both verify without a registry."""
    registry = Registry(tmp_path / "graph.sqlite3")
    m2a = build_m2a_symbolic_definition(project)
    registry.register_batch(m2a)
    old_root = m2a[-1]
    old_manifest = export_manifest(registry, old_root.id)
    new_root = snapshot_symbolic(project, registry)
    assert new_root != old_root
    old_nodes, new_nodes = _by_name(m2a), _by_name(build_symbolic_definition(project))
    for name in ("hybrid-miner", _MINING, "parser/semantic/schema.py"):
        assert old_nodes[name].id == new_nodes[name].id
    assert snapshot_symbolic(project, registry) == new_root
    assert registry.get(old_root.id) == old_root
    assert export_manifest(registry, old_root.id) == old_manifest
    new_manifest = export_manifest(registry, new_root.id)
    registry.path.unlink()
    for root, data in ((old_root, old_manifest), (new_root, new_manifest)):
        nodes = verify_manifest(data, expected_root_id=root.id)
        assert validate_symbolic_definition(nodes, root.id)[-1] == root


@pytest.fixture(scope="module")
def valid_definition() -> tuple[Node, ...]:
    return build_symbolic_definition(_REPO)


def _rewrite_semantic_node(
    nodes: tuple[Node, ...],
    name: str,
    edit: Callable[[dict[str, Any], list[Dependency]], None],
    extra: Node | None = None,
) -> tuple[Node, ...]:
    rewritten: dict[str, Node] = {}
    for node in nodes:
        payload = cast(dict[str, Any], node.to_dict()["payload"])
        edges = [
            Dependency(edge.role, edge.slot, rewritten[edge.target_id].id)
            for edge in node.dependencies
        ]
        if node.name == name:
            edit(payload, edges)
        rewritten[node.id] = Node(
            kind=node.kind,
            name=node.name,
            payload=payload,
            dependencies=tuple(edges),
        )
    indexed = {node.id: node for node in rewritten.values()}
    if extra is not None:
        indexed[extra.id] = extra
    return reachable_nodes((rewritten[nodes[-1].id].id,), indexed.__getitem__)


@pytest.mark.parametrize(
    ("case", "name", "error"),
    [
        ("projection", "prompt/router", "requires projection prompt-template"),
        (
            "version",
            "prompt/student",
            "requires projection prompt-instructions version 1",
        ),
        ("target", "grammar/atomese", "requires parser/grammar/atomese.py"),
        ("kind", "grammar/atomese", "Invalid M2b slots for grammar/atomese"),
        ("missing-builder", "prompt/student", "Invalid M2b slots for prompt/student"),
        (
            "missing-profile",
            "configuration/parser-backend",
            "Invalid M2b projection inputs",
        ),
        (
            "missing-guide",
            "prompt/discourse",
            "Missing nodes: .*prompt-guide/discourse",
        ),
        (
            "catalog-membership",
            "ontology/selected-catalog",
            "Missing nodes: .*module/extra",
        ),
        (
            "catalog-version",
            "ontology/selected-catalog",
            "version does not match module",
        ),
        ("unexpected", "ontology/module/general", "Unexpected nodes: .*ontology/extra"),
        (
            "split-loader",
            "ontology/miner-bundle",
            "requires a shared node for parser/ontology/catalog.py",
        ),
    ],
)
def test_rejects_rehashed_invalid_semantic_recipes(
    valid_definition: tuple[Node, ...], case: str, name: str, error: str
) -> None:
    nodes = _by_name(valid_definition)
    extra = None
    if case == "unexpected":
        extra = Node(
            kind="semantic_content",
            name="ontology/extra",
            payload={
                "projection": "prompt-guide",
                "version": 1,
                "value": {"guide": ""},
            },
        )
    elif case == "split-loader":
        loader = nodes["parser/ontology/catalog.py"]
        extra = Node(
            kind="source",
            name=loader.name,
            payload={**loader.payload, "sha256": "a" * 64},
        )

    def edit(payload: dict[str, Any], edges: list[Dependency]) -> None:
        if case == "projection":
            payload["projection"] = "ontology-module"
        elif case == "version":
            payload["version"] = 2
        elif case == "missing-profile":
            del payload["value"]["profile"]
        elif case == "missing-builder":
            edges.clear()
        elif case == "missing-guide":
            edges[:] = [edge for edge in edges if edge.slot != "catalog-guide"]
        elif case == "catalog-membership":
            payload["value"]["modules"].insert(1, "extra")
            payload["value"]["versions"].insert(1, "new-version")
        elif case == "catalog-version":
            payload["value"]["versions"][0] = "changed-version"
        elif case == "target":
            edges[:] = [edge for edge in edges if edge.slot != "grammar-loader"]
            edges.append(
                Dependency(
                    "uses-source",
                    "grammar-loader",
                    nodes["parser/ontology/catalog.py"].id,
                )
            )
        elif case == "kind":
            edges[:] = [edge for edge in edges if edge.slot != "grammar-loader"]
            edges.append(
                Dependency(
                    "uses-semantic",
                    "grammar-loader",
                    nodes["ontology/default-vocabulary"].id,
                )
            )
        elif case == "unexpected":
            assert extra is not None
            edges.append(Dependency("uses-semantic", "extra", extra.id))
        elif case == "split-loader":
            assert extra is not None
            edges[:] = [Dependency("uses-source", "catalog-loader", extra.id)]

    altered = _rewrite_semantic_node(valid_definition, name, edit, extra)
    root_id = altered[-1].id
    assert root_id != valid_definition[-1].id
    manifest = canonical_json(
        {
            "manifest_version": MANIFEST_VERSION,
            "root_id": root_id,
            "nodes": [node.to_dict() for node in sorted(altered, key=lambda n: n.id)],
        }
    )
    # A fully rehashed graph can pass generic verification and still violate M2b.
    verified = verify_manifest(manifest, expected_root_id=root_id)
    with pytest.raises(ValueError, match=error):
        validate_symbolic_definition(verified, root_id)


def test_distilled_profile_is_outside_reference_definition(project: Path) -> None:
    before = build_symbolic_definition(project)
    config = project / _CONFIG / "model_backend.yaml"
    document = _load(config)
    inactive_student = document["profiles"]["semantic_document_student"]
    inactive_student.update(
        model="models/another-student", device="cpu", max_new_tokens=4096
    )
    _write(config, document)
    assert build_symbolic_definition(project) == before

    # Selecting it as active makes this a Transformers reference backend.
    document["active"] = "semantic_document_student"
    _write(config, document)
    active = backend_projection(config)
    assert active["student_role"] is True
    assert active["model"] == "models/another-student"
    assert active["max_new_tokens"] == 4096
    assert active["add_special_tokens"] is False
    inactive_student["device"] = "cuda"
    _write(config, document)
    assert backend_projection(config) == active
    inactive_student["add_special_tokens"] = True
    _write(config, document)
    assert backend_projection(config)["add_special_tokens"] is True


def test_reference_profile_records_variable_names_without_reading_environment(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = project / _CONFIG / "model_backend.yaml"
    document = {
        "active": "fixture",
        "profiles": {
            "fixture": {
                "provider": "openai-compatible",
                "model_env": "M2B_TEST_MODEL",
                "base_url_env": "M2B_TEST_HOST",
            }
        },
    }
    _write(config, document)
    monkeypatch.delenv("M2B_TEST_MODEL", raising=False)
    monkeypatch.delenv("M2B_TEST_HOST", raising=False)
    before = backend_projection(config)
    assert before["model_env"] == "M2B_TEST_MODEL"
    assert before["base_url_env"] == "M2B_TEST_HOST"
    monkeypatch.setenv("M2B_TEST_MODEL", "fixture-model")
    monkeypatch.setenv("M2B_TEST_HOST", "fixture-host")
    assert backend_projection(config) == before


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("parser/semantic/schema.py"),
        Path("pyproject.toml"),
        _GENERAL,
        _BUNDLE,
        _CONFIG / "model_backend.yaml",
        Path("alternative.yaml"),
    ],
)
def test_snapshot_rejects_changed_input_bytes_before_registration(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
) -> None:
    alternate = project / "alternative.yaml"
    copyfile(project / _BUNDLE, alternate)
    path = project / relative_path
    read_bytes = Path.read_bytes
    edited = False

    def edit_during_capture(current: Path) -> bytes:
        nonlocal edited
        captured = read_bytes(current)
        if current == path and not edited:
            edited = True
            path.write_bytes(captured + b"\n# changed during capture\n")
        return captured

    monkeypatch.setattr(Path, "read_bytes", edit_during_capture)
    registry = Registry(tmp_path / "graph.sqlite3")
    with pytest.raises(ValueError, match="snapshot input changed during capture"):
        snapshot_symbolic(project, registry, ontology_bundle=alternate)
    assert edited
    assert not registry.path.exists()


def test_snapshot_rejects_ontology_directory_membership_change(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = project / _CONFIG / "model_backend.yaml"
    added = project / _ONTOLOGY / "added.yaml"
    read_bytes = Path.read_bytes

    def add_module_during_capture(path: Path) -> bytes:
        if path == backend and not added.exists():
            added.write_text("# added during capture\n", encoding="utf-8")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", add_module_during_capture)
    registry = Registry(tmp_path / "graph.sqlite3")
    with pytest.raises(ValueError, match="ontology module set changed during capture"):
        snapshot_symbolic(project, registry)
    assert not registry.path.exists()
