import json
import sqlite3
import subprocess
import sys
from collections.abc import Iterable
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import cast

import pytest
import tomllib

from versioning.manifest import export_manifest, verify_manifest
from versioning.nodes import Dependency, Node
from versioning.registry import Registry
from versioning.symbolic import (
    build_m2a_symbolic_definition,
    read_hybrid_miner_release,
    validate_symbolic_definition,
)
from versioning.symbolic import (
    snapshot_symbolic as _snapshot_symbolic,
)

build_symbolic_definition = build_m2a_symbolic_definition
snapshot_symbolic = partial(_snapshot_symbolic, recipe="m2a")

_CWD = Path(__file__).resolve().parents[2]
_ORIGIN = "https://github.com/iCog-Labs-Dev/hybrid-miner.git"
_PARSER = "symbolic/parser"
_ADAPTER = "symbolic/adapter"
_MINING = "symbolic/mining"
_ROOT = "symbolic/definition"
_RELEASE = "hybrid-miner"
_PARSER_SOURCE = "parser/semantic/semantic_parser.py"
_CONTRACT = "parser/semantic/schema.py"
_ADAPTER_SOURCE = "symbolic_pipeline/miner_adapter.py"
_SOURCES = {
    _PARSER_SOURCE,
    _CONTRACT,
    _ADAPTER_SOURCE,
    "symbolic_pipeline/miner_bundle.py",
    "symbolic_pipeline/document_export.py",
}


def _write_requirements(pyproject: Path, *requirements: str) -> Path:
    pyproject.write_text(
        "[project.optional-dependencies]\n" f"miner = {json.dumps(requirements)}\n",
        encoding="utf-8",
    )
    return pyproject


def _requirement(selector: str) -> str:
    return f"{_RELEASE} @ git+{_ORIGIN}@{selector}"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for name in sorted(_SOURCES):
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((_CWD / name).read_bytes())
    _write_requirements(root / "pyproject.toml", _requirement("v0.1.0"))
    return root


def _by_name(nodes: Iterable[Node]) -> dict[str, Node]:
    return {node.name: node for node in nodes}


def test_real_checked_in_declaration_and_source_inventory() -> None:
    nodes = _by_name(build_symbolic_definition(_CWD))
    leaf = nodes[_RELEASE]
    assert leaf.payload == {
        "repository": _ORIGIN,
        "selector": {"kind": "tag", "value": "v0.1.1"},
    }
    assert leaf.kind == "external_release"
    assert set(nodes) == {*_SOURCES, _RELEASE, _PARSER, _ADAPTER, _MINING, _ROOT}
    assert {edge.target_id for edge in nodes[_PARSER].dependencies} == {
        nodes[_PARSER_SOURCE].id,
        nodes[_CONTRACT].id,
    }
    assert {
        edge.target_id
        for edge in nodes[_ADAPTER].dependencies
        if edge.slot == "parser-contract"
    } == {nodes[_CONTRACT].id}


@pytest.mark.parametrize(
    ("selector", "kind", "value"),
    [
        ("v0.1.1", "tag", "v0.1.1"),
        ("1.2.3-rc.1", "tag", "1.2.3-rc.1"),
        ("refs/tags/releases/production-7", "tag", "releases/production-7"),
        ("A1" * 20, "commit", "a1" * 20),
        ("2B" * 32, "commit", "2b" * 32),
    ],
)
def test_git_selectors(tmp_path: Path, selector: str, kind: str, value: str) -> None:
    pyproject = _write_requirements(
        tmp_path / "pyproject.toml", "other-package>=2", _requirement(selector)
    )
    leaf = read_hybrid_miner_release(pyproject)
    assert leaf.name == _RELEASE
    assert leaf.payload["repository"] == _ORIGIN
    assert leaf.payload["selector"] == {"kind": kind, "value": value}


def test_equivalent_locator_and_package_spelling_keep_identity(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    _write_requirements(path, _requirement("v0.1.1"))
    original = read_hybrid_miner_release(path)
    equivalent = (
        "Hybrid_Miner @ git+https://GITHUB.COM/icog-labs-dev/HYBRID-MINER"
        "@refs/tags/v0.1.1"
    )
    actual = read_hybrid_miner_release(_write_requirements(path, equivalent))
    assert actual == original


@pytest.mark.parametrize(
    ("requirements", "message"),
    [
        ((), "Missing hybrid-miner"),
        (("other-package>=2",), "Missing hybrid-miner"),
        ((_requirement("v0.1.1"), "Hybrid.Miner>=1"), "Duplicate hybrid-miner"),
        (("hybrid-miner>=1",), "exact Git direct requirement"),
        (("hybrid-miner[extra] @ git+" + _ORIGIN + "@v0.1.1",), "direct requirement"),
        (("hybrid-miner @ " + _ORIGIN + "@v0.1.1",), r"git\+https"),
        (
            (_requirement("v0.1.1").replace("git+https://", "git+ssh://"),),
            r"git\+https",
        ),
        (
            (_requirement("v0.1.1").replace("iCog-Labs-Dev", "different-owner"),),
            "Unexpected hybrid-miner repository",
        ),
        (
            (_requirement("v0.1.1").replace("github.com", "user@github.com"),),
            "Unexpected hybrid-miner repository",
        ),
        (("hybrid-miner @ git+" + _ORIGIN,), "exact Git selector"),
        ((_requirement(""),), "version tag"),
        ((_requirement("main"),), "version tag"),
        ((_requirement("HEAD"),), "version tag"),
        ((_requirement("refs/heads/v0.1.1"),), "version tag"),
        ((_requirement("abc1234"),), "full Git commit"),
        ((_requirement("v0.1.*"),), "version tag"),
        ((_requirement("v0.1.1#subdirectory=python"),), "version tag"),
        ((_requirement('v0.1.1 ; python_version >= "3.11"'),), "version tag"),
        ((_requirement("refs/tags/latest"),), "exact release tag"),
        ((_requirement("refs/tags/invalid..tag"),), "Git tag name"),
    ],
)
def test_rejects_invalid_declarations(
    tmp_path: Path, requirements: tuple[str, ...], message: str
) -> None:
    path = _write_requirements(tmp_path / "pyproject.toml", *requirements)
    with pytest.raises(ValueError, match=message):
        read_hybrid_miner_release(path)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("[project]\n", "Missing"),
        (
            "[project.optional-dependencies]\nminer = [1]\n",
            "must contain requirement strings",
        ),
    ],
)
def test_rejects_missing_or_invalid_miner_extra(
    tmp_path: Path, data: str, message: str
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_hybrid_miner_release(path)


def test_rejects_malformed_toml(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text("[project.optional-dependencies]\nminer = [", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        read_hybrid_miner_release(path)


def _changed(old: dict[str, Node], new: dict[str, Node]) -> set[str]:
    assert old.keys() == new.keys()
    return {key for key in old if old[key].id != new[key].id}


def test_release_propagation_historical_roots_and_manifest(
    project: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    old = snapshot_symbolic(project, registry)
    first = export_manifest(registry, old.id)
    old_nodes = _by_name(verify_manifest(first))

    _write_requirements(project / "pyproject.toml", _requirement("v0.1.1"))
    new = snapshot_symbolic(project, registry)
    final = export_manifest(registry, new.id)
    verified = verify_manifest(final, expected_root_id=new.id)
    new_nodes = _by_name(verified)
    assert _changed(old_nodes, new_nodes) == {_RELEASE, _ADAPTER, _MINING, _ROOT}
    release = new_nodes[_RELEASE]
    assert release.payload["selector"] == {"kind": "tag", "value": "v0.1.1"}
    for stage in (_ADAPTER, _MINING):
        assert [
            edge.target_id
            for edge in new_nodes[stage].dependencies
            if edge.role == "uses-release"
        ] == [release.id]
    assert registry.get(old.id) == old
    assert registry.get(new.id) == new
    assert export_manifest(registry, old.id) == first

    registry.path.unlink()
    assert validate_symbolic_definition(verify_manifest(final), new.id)[-1] == new


def test_unrelated_configuration_and_repeat_do_not_change_root(
    project: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    root = snapshot_symbolic(project, registry)
    assert snapshot_symbolic(project, registry) == root
    with (project / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[tool.unrelated]\nversion = "changed"\n')
    assert snapshot_symbolic(project, registry) == root
    with closing(sqlite3.connect(registry.path)) as database:
        assert database.execute("SELECT count(*) FROM nodes").fetchone() == (10,)


@pytest.mark.parametrize(
    ("file", "ancestors"),
    [
        (_PARSER_SOURCE, {_PARSER, _ROOT}),
        (_ADAPTER_SOURCE, {_ADAPTER, _ROOT}),
        (_CONTRACT, {_PARSER, _ADAPTER, _ROOT}),
    ],
)
def test_source_edit_changes_only_its_ancestors(
    project: Path, file: str, ancestors: set[str]
) -> None:
    before = _by_name(build_symbolic_definition(project))
    with (project / file).open("ab") as stream:
        stream.write(b"\n# changed in fixture\n")
    after = _by_name(build_symbolic_definition(project))
    assert _changed(before, after) == {file, *ancestors}


def _with_edges(node: Node, edges: Iterable[Dependency]) -> Node:
    return Node(
        kind=node.kind,
        name=node.name,
        payload=cast(dict[str, object], node.to_dict()["payload"]),
        dependencies=tuple(edges),
    )


@pytest.mark.parametrize(
    ("slot", "role", "target", "message"),
    [
        ("parser", None, None, "Missing slots"),
        ("adapter", "uses-component", _PARSER, "requires symbolic/adapter"),
        ("mining", "uses-component", _PARSER_SOURCE, "requires composite, got source"),
    ],
)
def test_validates_required_slots_and_target_kinds(
    project: Path, slot: str, role: str | None, target: str | None, message: str
) -> None:
    nodes = _by_name(build_symbolic_definition(project))
    root = nodes[_ROOT]
    edges = [edge for edge in root.dependencies if edge.slot != slot]
    if role is not None and target is not None:
        edges.append(Dependency(role, slot, nodes[target].id))
    wrong = _with_edges(root, edges)
    with pytest.raises(ValueError, match=message):
        validate_symbolic_definition([*nodes.values(), wrong], wrong.id)


def test_recipe_rejects_different_declared_leaves_for_the_two_consumers(
    project: Path,
) -> None:
    nodes = _by_name(build_symbolic_definition(project))
    pyproject = _write_requirements(project / "pyproject.toml", _requirement("v0.1.1"))
    other_release = read_hybrid_miner_release(pyproject)
    mining = _with_edges(
        nodes[_MINING], (Dependency("uses-release", _RELEASE, other_release.id),)
    )
    root = _with_edges(
        nodes[_ROOT],
        (
            (
                Dependency(edge.role, edge.slot, mining.id)
                if edge.slot == "mining"
                else edge
            )
            for edge in nodes[_ROOT].dependencies
        ),
    )
    with pytest.raises(ValueError, match="requires a shared node for hybrid-miner"):
        validate_symbolic_definition(
            [*nodes.values(), other_release, mining, root], root.id
        )


def test_snapshot_rolls_back_all_new_nodes_on_insert_failure(
    project: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    old = snapshot_symbolic(project, registry)
    _write_requirements(project / "pyproject.toml", _requirement("v0.1.1"))
    nodes = _by_name(build_symbolic_definition(project))
    with closing(sqlite3.connect(registry.path)) as database, database:
        database.execute(
            f"""
            CREATE TRIGGER reject_definition BEFORE INSERT ON nodes
            WHEN NEW.id = '{nodes[_ROOT].id}'
            BEGIN SELECT RAISE(ABORT, 'fixture write failure'); END
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="fixture write failure"):
        snapshot_symbolic(project, registry)
    assert registry.get(old.id) == old
    for name in (_RELEASE, _ADAPTER, _MINING, _ROOT):
        with pytest.raises(KeyError):
            registry.get(nodes[name].id)


def test_cli_prints_only_a_stable_root_id(project: Path, tmp_path: Path) -> None:
    database = tmp_path / "graph.sqlite3"
    command = [
        sys.executable,
        "-B",
        "-S",
        "-m",
        "versioning",
        "snapshot-symbolic",
        "--recipe",
        "m2a",
        "--project-root",
        str(project),
        "--registry",
        str(database),
    ]
    expected = build_symbolic_definition(project)[-1]
    for _ in range(2):
        result = subprocess.run(command, cwd=_CWD, capture_output=True, check=True)
        assert result.stdout == f"{expected.id}\n".encode()
        assert result.stderr == b""
    assert Registry(database).get(expected.id) == expected
