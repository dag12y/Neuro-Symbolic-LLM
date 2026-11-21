import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier, Event
from typing import Any, cast

import pytest

from versioning.hashing import canonical_json, load_json
from versioning.manifest import export_manifest, verify_manifest
from versioning.nodes import (
    Dependency,
    DraftDependency,
    DraftNode,
    Node,
    build_nodes,
    reachable_nodes,
)
from versioning.registry import Registry


def _composite(purpose: str = "component") -> dict[str, object]:
    return {"recipe": {"name": "fixture/symbolic", "version": 1}, "purpose": purpose}


def _source(name: str) -> Node:
    return Node(
        kind="source", name=name, payload={"sha256": "a" * 64, "byte_length": 2}
    )


def _graph(tag: str = "v0.1.0", *, reverse: bool = False) -> dict[str, Node]:
    stages = ("parser", "adapter", "mining")
    if reverse:
        stages = stages[::-1]
    drafts = {
        "root": DraftNode(
            kind="composite",
            name="symbolic",
            payload=_composite("definition"),
            dependencies=tuple(
                DraftDependency("uses-component", stage, stage)
                for stage in stages
            ),
        ),
        "parser": DraftNode(
            kind="composite",
            name="symbolic/parser",
            payload=_composite(),
            dependencies=(DraftDependency("uses-source", "implementation", "source"),),
        ),
        "adapter": DraftNode(
            kind="composite",
            name="symbolic/adapter",
            payload=_composite(),
            dependencies=(DraftDependency("uses-release", "miner", "release"),),
        ),
        "mining": DraftNode(
            kind="composite",
            name="symbolic/mining",
            payload=_composite(),
            dependencies=(DraftDependency("uses-release", "miner", "release"),),
        ),
        "source": DraftNode(
            kind="source",
            name="parser/example.py",
            payload={"sha256": "a" * 64, "byte_length": 2},
        ),
        "release": DraftNode(
            kind="external_release",
            name="hybrid-miner",
            payload={
                "repository": "https://github.com/iCog-Labs-Dev/hybrid-miner.git",
                "selector": {"kind": "tag", "value": tag},
            },
        ),
    }
    return build_nodes(dict(reversed(list(drafts.items()))) if reverse else drafts)


def test_node_order_and_shared_dependencies() -> None:
    nodes = _graph()
    assert _graph(reverse=True) == nodes
    assert [edge.slot for edge in nodes["root"].dependencies] == [
        "adapter",
        "mining",
        "parser",
    ]
    assert nodes["adapter"].dependencies == nodes["mining"].dependencies
    assert nodes["adapter"].dependencies[0].target_id == nodes["release"].id
    indexed = {node.id: node for node in nodes.values()}
    ordered = reachable_nodes((nodes["root"].id,), indexed.__getitem__)
    assert len(ordered) == len(indexed)
    assert ordered[-1] == nodes["root"]
    for node in ordered:
        assert Node.from_dict(node.to_dict()) == node


def test_nested_payload_is_an_immutable_snapshot() -> None:
    repo = "https://github.com/iCog-Labs-Dev/hybrid-miner.git"
    selector = {"kind": "tag", "value": "v0.1.1"}
    node = Node(
        kind="external_release",
        name="hybrid-miner",
        payload={"repository": repo, "selector": selector},
    )
    selector["value"] = "v0.1.2"
    frozen_selector = cast(Any, node.payload["selector"])
    with pytest.raises(TypeError):
        frozen_selector["value"] = "v0.1.3"
    snapshot = cast(dict[str, Any], node.to_dict())
    snapshot["payload"]["selector"]["value"] = "v0.1.4"
    assert frozen_selector["value"] == "v0.1.1"
    assert Node.from_dict(node.to_dict()).id == node.id


def test_source_name_is_a_portable_logical_path() -> None:
    node = _source(".config/résumé.md")
    assert Node.from_dict(node.to_dict()).name == node.name
    for name in ("C:/checkout/source.py", "src/../source.py", "src//source.py"):
        with pytest.raises(ValueError, match="relative"):
            _source(name)


@pytest.mark.parametrize(
    "kind, name, payload, match",
    [
        (
            "source",
            "/home/user/source.py",
            {"sha256": "a" * 64, "byte_length": 2},
            "relative",
        ),
        (
            "source",
            "source.py",
            {"sha256": "a" * 64, "byte_length": True},
            "byte_length",
        ),
        ("source", "source.py", {"sha256": "a" * 64, "byte_length": -1}, "byte_length"),
        (
            "source",
            "source.py",
            {"sha256": "a" * 64, "byte_length": 0, "path": "/tmp"},
            "Source payload",
        ),
        (
            "external_release",
            "miner",
            {"repository": "/tmp/miner", "selector": {"kind": "tag", "value": "v1"}},
            "HTTPS",
        ),
        (
            "external_release",
            "miner",
            {
                "repository": "https://user@host/miner",
                "selector": {"kind": "tag", "value": "v1"},
            },
            "HTTPS",
        ),
        (
            "external_release",
            "miner",
            {
                "repository": "https://host/miner",
                "selector": {"kind": "commit", "value": "abcd"},
            },
            "full lowercase",
        ),
        (
            "composite",
            "root",
            {"recipe": {"name": "test", "version": 0}, "purpose": "definition"},
            "positive",
        ),
        ("composite", "root", _composite("unknown"), "purpose"),
    ],
)
def test_invalid_payloads(
    kind: str, name: str, payload: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Node(kind=kind, name=name, payload=payload)


def test_duplicate_slots_and_wrong_target_kind() -> None:
    source = _source("one.py")
    edge = Dependency("uses-source", "impl", source.id)
    with pytest.raises(ValueError, match="Duplicate dependency slot: uses-source/impl"):
        Node(
            kind="composite",
            name="parent",
            payload=_composite(),
            dependencies=(edge, edge),
        )

    wrong_edge = Dependency("uses-release", "miner", source.id)
    node = Node(
        kind="composite",
        name="parent",
        payload=_composite(),
        dependencies=(wrong_edge,),
    )
    indexed = {node.id: node, source.id: source}
    with pytest.raises(ValueError, match="requires external_release, got source"):
        reachable_nodes((node.id,), indexed.__getitem__)


def _one_draft(name: str, target: str) -> DraftNode:
    return DraftNode(
        kind="composite",
        name=name,
        payload=_composite(),
        dependencies=(DraftDependency("uses-component", "prerequisite", target),),
    )


def test_draft_missing_reference_and_cycle() -> None:
    with pytest.raises(ValueError, match="Missing dependency node: missing"):
        build_nodes({"one": _one_draft("one", "missing")})
    with pytest.raises(ValueError, match="Dependency cycle.*one -> two -> one"):
        build_nodes({"two": _one_draft("two", "one"), "one": _one_draft("one", "two")})


def test_resolved_graph_detects_cycles() -> None:
    # Use synthetic IDs to test graph validation independently of digest checks.
    ids = ("1" * 64, "2" * 64)
    nodes = {}
    for key, target in (ids, ids[::-1]):
        node = Node(
            kind="composite",
            name="cycle",
            payload=_composite(),
            dependencies=(Dependency("uses-component", "loop", target),),
        )
        object.__setattr__(node, "id", key)
        nodes[key] = node
    with pytest.raises(ValueError, match="Dependency cycle at slot loop"):
        reachable_nodes((ids[0],), nodes.__getitem__)


def test_adopted_release_changes_only_its_ancestors(tmp_path: Path) -> None:
    old = _graph("v0.1.0")
    new = _graph("v0.1.1")
    assert {name for name in old if old[name].id != new[name].id} == {
        "release",
        "adapter",
        "mining",
        "root",
    }

    registry = Registry(tmp_path / "graph.sqlite3")
    for version in (old, new):
        ids = registry.register_batch(reversed(tuple(version.values())))
        assert set(ids) == {node.id for node in version.values()}
    for node in (*old.values(), *new.values()):
        assert registry.get(node.id) == node
    old_manifest = export_manifest(registry, old["root"].id)
    assert len(verify_manifest(old_manifest)) == len(old)
    assert verify_manifest(old_manifest)[-1] == old["root"]


def test_registry_reuses_existing_nodes(tmp_path: Path) -> None:
    nodes = _graph()
    registry = Registry(tmp_path / "graph #?.sqlite3")
    children = [node for key, node in nodes.items() if key != "root"]
    registry.register_batch(children)
    root = nodes["root"]
    for _ in range(2):
        assert registry.register_batch((root, root)) == (root.id,)
        assert registry.get(root.id) == root
    with closing(sqlite3.connect(registry.path)) as database:
        count = database.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert count == len(nodes)


def test_registry_missing_target_rejects_entire_batch(tmp_path: Path) -> None:
    nodes = _graph()
    registry = Registry(tmp_path / "graph.sqlite3")
    with pytest.raises(ValueError, match="Missing dependency node"):
        registry.register_batch((nodes["source"], nodes["root"]))
    with pytest.raises(KeyError):
        registry.get(nodes["source"].id)


def test_sql_failure_rolls_back_prior_inserts(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    first, last, existing = map(_source, ("first", "last", "existing"))
    registry.register_batch((existing,))
    with closing(sqlite3.connect(registry.path)) as database, database:
        database.execute(
            f"""
            CREATE TRIGGER reject_insert BEFORE INSERT ON nodes
            WHEN NEW.id = '{last.id}'
            BEGIN SELECT RAISE(ABORT, 'injected insertion failure'); END
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected insertion failure"):
        registry.register_batch((first, last))
    assert registry.get(existing.id) == existing
    for node in (first, last):
        with pytest.raises(KeyError):
            registry.get(node.id)


def test_registry_rejects_conflicting_stored_bytes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    node = _source("source.py")
    registry.register_batch((node,))
    with closing(sqlite3.connect(registry.path)) as database, database:
        database.execute(
            "UPDATE nodes SET canonical = ? WHERE id = ?", (b"{}", node.id)
        )
    with pytest.raises(ValueError, match="Conflicting data"):
        registry.register_batch((_source("other.py"), node))
    with pytest.raises(KeyError):
        registry.get(_source("other.py").id)
    with pytest.raises(ValueError, match="Node must contain exactly"):
        registry.get(node.id)


def test_concurrent_registration_of_overlapping_batches(tmp_path: Path) -> None:
    database = tmp_path / "graph.sqlite3"
    old, new = _graph("v0.1.0"), _graph("v0.1.1")
    start = Barrier(3, timeout=5)

    def register(nodes: dict[str, Node]) -> None:
        start.wait()
        Registry(database).register_batch(nodes.values())

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(register, nodes) for nodes in (old, new, new)]
        for future in futures:
            future.result(timeout=10)
    with closing(sqlite3.connect(database)) as connection:
        ids = {row[0] for row in connection.execute("SELECT id FROM nodes")}
    assert ids == {node.id for node in (*old.values(), *new.values())}


def test_reader_cannot_see_uncommitted_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    existing, first, last = map(_source, ("existing", "first", "last"))
    registry.register_batch((existing,))
    connect = sqlite3.connect
    writer_paused, reader_done = Event(), Event()

    def pause_write() -> int:
        writer_paused.set()
        if not reader_done.wait(timeout=5):
            raise TimeoutError("reader did not finish")
        return 0

    def connect_with_pause(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)
        connection.create_function("pause_write", 0, pause_write)
        return connection

    with closing(connect(registry.path)) as database, database:
        database.execute(
            f"""
            CREATE TRIGGER pause_last_insert BEFORE INSERT ON nodes
            WHEN NEW.id = '{last.id}'
            BEGIN SELECT pause_write(); END
            """
        )
    monkeypatch.setattr(sqlite3, "connect", connect_with_pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(registry.register_batch, (first, last))
        try:
            assert writer_paused.wait(timeout=5)
            assert registry.get(existing.id) == existing
            with pytest.raises(KeyError):
                registry.get(first.id)
        finally:
            reader_done.set()
        assert writer.result(timeout=5) == (first.id, last.id)
    assert registry.get(first.id) == first
    assert registry.get(last.id) == last


def _manifest(tmp_path: Path) -> tuple[Registry, bytes, Node]:
    registry = Registry(tmp_path / "graph.sqlite3")
    nodes = _graph()
    registry.register_batch((*nodes.values(), _source("unrelated.py")))
    root = nodes["root"]
    return registry, export_manifest(registry, root.id), root


def test_offline_manifest_round_trip_and_expected_root(tmp_path: Path) -> None:
    registry, data, root = _manifest(tmp_path)
    registry.path.unlink()
    verified = verify_manifest(data, expected_root_id=root.id)
    assert len(verified) == 6
    assert verified[-1] == root
    manifest = cast(dict[str, Any], load_json(data))
    assert sorted(node.id for node in verified) == [
        item["id"] for item in manifest["nodes"]
    ]
    with pytest.raises(ValueError, match="does not match the expected root"):
        verify_manifest(data, expected_root_id=_source("other.py").id)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "Duplicate manifest node"),
        ("missing", "Missing dependency node"),
        ("missing-root", "Missing dependency node"),
        ("unreachable", "unreachable"),
        ("tamper-id", "Node ID does not match"),
        ("tamper-release", "Node ID does not match"),
        ("payload", "Release payload"),
        ("version", "Unsupported manifest version"),
        ("edge-order", "dependencies are out of canonical order"),
        ("node-order", "nodes are out of ID order"),
    ],
)
def test_manifest_rejects_corruption(tmp_path: Path, mutation: str, match: str) -> None:
    _, data, _ = _manifest(tmp_path)
    document = cast(dict[str, Any], load_json(data))
    nodes = document["nodes"]
    release = next(node for node in nodes if node["kind"] == "external_release")
    if mutation == "duplicate":
        nodes.append(release)
    elif mutation == "missing":
        nodes.remove(release)
    elif mutation == "missing-root":
        document["root_id"] = "0" * 64
    elif mutation == "unreachable":
        nodes.append(_source("wrong.py").to_dict())
        nodes.sort(key=lambda node: node["id"])
    elif mutation == "tamper-id":
        release["id"] = "0" * 64
    elif mutation == "tamper-release":
        release["payload"]["selector"]["value"] = "v0.1.1"
    elif mutation == "payload":
        release["payload"]["verified"] = True
    elif mutation == "version":
        document["manifest_version"] = 1
    elif mutation == "edge-order":
        root = next(node for node in nodes if node["id"] == document["root_id"])
        root["dependencies"].reverse()
    else:
        nodes.reverse()
    with pytest.raises(ValueError, match=match):
        verify_manifest(canonical_json(document))


def test_cli_export_and_offline_verification(tmp_path: Path) -> None:
    registry, data, root = _manifest(tmp_path)

    def run(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-B", "-m", "versioning", *args],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
        )

    inspected = run("inspect", "--registry", str(registry.path), root.id)
    assert inspected.returncode == 0, inspected.stderr
    assert load_json(inspected.stdout) == root.to_dict()
    exported = run("export-manifest", "--registry", str(registry.path), root.id)
    assert exported.returncode == 0, exported.stderr
    assert exported.stdout == data + b"\n"

    registry.path.unlink()
    path = tmp_path / "published.json"
    path.write_bytes(exported.stdout)
    success = run("verify-manifest", str(path), "--expected-root", root.id)
    assert success.returncode == 0, success.stderr
    assert b"Verified graph" in success.stdout
    failure = run("verify-manifest", str(path), "--expected-root", "0" * 64)
    assert failure.returncode == 1
    assert b"does not match the expected root" in failure.stderr
