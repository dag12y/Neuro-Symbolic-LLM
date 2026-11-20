"""Offline tests for component identities, local storage, and manifests."""

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Barrier, Event

import pytest

from versioning import (
    CompatibilityResult,
    CompatibilityStatus,
    ComponentRecord,
    Dependency,
    Registry,
    canonical_json,
    export_manifest,
    hash_directory,
    hash_file,
    hash_identity,
    load_json,
    verify_manifest,
)
from versioning.records import dependency_order


def component(**fields):
    defaults = {
        "kind": "prompt",
        "name": "extract",
        "content_digest": hashlib.sha256(b"extract facts").hexdigest(),
        "digest_scope": "raw file bytes",
    }
    return ComponentRecord(**(defaults | fields))


def test_canonical_json_and_stable_ids():
    left = component(details={"z": [1, 2], "a": "é"})
    right = component(details={"a": "é", "z": [1, 2]})
    payload = b'{"a":"\\u00e9","z":[1,2]}'
    assert canonical_json(right.to_dict()["details"]) == payload
    assert hash_identity(right.to_dict()["details"]) == hashlib.sha256(
        b"versioning:component:v1\0" + payload
    ).hexdigest()
    assert left.record_id == right.record_id
    assert ComponentRecord.from_dict(right.to_dict()) == right


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "ontology"},
        {"name": "other"},
        {"content_digest": "0" * 64},
        {"digest_scope": "directory tree v1"},
        {"label": "v1", "label_role": "family"},
        {"details": {"array": [2, 1]}},
        {"producer": {"tool": "synthetic"}},
        {"dependencies": (Dependency("schema", "0" * 64),)},
    ],
)
def test_identity_changes(change):
    before = component(details={"array": [1, 2]})
    assert replace(before, **change).record_id != before.record_id


def test_file_and_directory_hashing(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    data = b"x" * (2**20)
    (first / "b").write_bytes(data)
    (first / "empty").mkdir()
    (second / "empty").mkdir()
    (second / "b").write_bytes(data)
    os.utime(second / "b", (10, 10))
    assert hash_file(first / "b") == hash_file(second / "b")
    assert hash_file(first / "b") == hashlib.sha256(data).hexdigest()
    assert hash_directory(first) == hash_directory(second)
    (second / "empty").rmdir()
    assert hash_directory(first) != hash_directory(second)
    (second / "empty").mkdir()
    (second / "b").rename(second / "renamed")
    assert hash_directory(first) != hash_directory(second)


def test_directory_rejects_symlinks_and_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    with pytest.raises(OSError):
        hash_directory(alias)
    (root / "escape").symlink_to(target)
    with pytest.raises(OSError):
        hash_directory(root)
    with pytest.raises(ValueError, match="must not contain"):
        hash_directory(root / ".." / target.name)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), {1: "value"}, (1,), b"x"]
)
def test_invalid_json_values(value):
    with pytest.raises(ValueError):
        canonical_json({"nested": [value]})


@pytest.mark.parametrize(
    "data",
    [
        '{"x":}',
        '{"x":{"a":1,"a":2}}',
        "[NaN]",
        "[Infinity]",
        "[-Infinity]",
        "[1e900]",
        r'["\ud800"]',
    ],
)
def test_invalid_serialized_json(data):
    with pytest.raises(ValueError):
        load_json(data)


def test_deep_immutability():
    details = {"items": [{"n": 1}]}
    producer = {"source": {"revision": "a"}}
    record = component(details=details, producer=producer)
    expected = record.to_dict()
    details["items"][0]["n"] = 2
    producer["source"]["revision"] = "b"
    copy = record.to_dict()
    copy["details"]["items"][0]["n"] = 3
    assert record.to_dict() == expected
    with pytest.raises(TypeError):
        record.details["items"][0]["n"] = 4
    with pytest.raises(TypeError):
        record.producer["source"]["revision"] = "c"
    with pytest.raises(FrozenInstanceError):
        record.name = "other"


@pytest.mark.parametrize(
    "fields",
    [
        {"label": "v1"},
        {"label_role": "release"},
        {"label_role": "unexpected"},
        {"digest_scope": ""},
        {"content_digest": "0" * 63},
        {"dependencies": ("invalid",)},
    ],
)
def test_invalid_record_fields(fields):
    with pytest.raises(ValueError):
        component(**fields)


def test_registration_lookup_and_no_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry = Registry()
    record = component()
    assert not (tmp_path / ".git").exists()
    record_id = registry.register(record)
    path = registry.root / "records" / f"{record_id}.json"
    inode = path.stat().st_ino
    assert registry.register(record) == record_id
    assert path.stat().st_ino == inode
    assert registry.get(record_id) == record
    assert registry.find("prompt", "extract") == [record]
    with pytest.raises(FileNotFoundError):
        registry.get("0" * 64)
    with pytest.raises(ValueError):
        registry.get("../outside")


def test_release_and_family_labels(tmp_path):
    registry = Registry(tmp_path)
    release = component(label="v1", label_role="release")
    registry.register(release)
    with pytest.raises(ValueError, match="Conflicting release label"):
        registry.register(replace(release, content_digest="0" * 64))
    registry.register(replace(release, name="another"))
    family = component(label="stable", label_role="family")
    other = replace(family, content_digest="0" * 64)
    registry.register(family)
    registry.register(other)
    assert {r.record_id for r in registry.find("prompt", "extract", "stable")} == {
        family.record_id,
        other.record_id,
    }
    assert registry.find("prompt", "extract", "v1") == [release]


def test_missing_dependency_and_graph_cycle(tmp_path):
    registry = Registry(tmp_path)
    leaf = component(name="leaf")
    root = component(dependencies=(Dependency("input", leaf.record_id),))
    with pytest.raises(ValueError, match="Missing dependency"):
        registry.register(root)
    assert registry.find("prompt", "extract") == []

    # A genuine hash-verified cycle would require a fixed point. Exercise the
    # graph validator directly with synthetic node IDs.
    first, second = "0" * 64, "1" * 64
    with pytest.raises(ValueError, match="Dependency cycle"):
        dependency_order(
            first,
            {
                first: component(dependencies=(Dependency("input", second),)),
                second: component(dependencies=(Dependency("input", first),)),
            },
        )


def test_corrupted_record_detection(tmp_path):
    registry = Registry(tmp_path)
    record = component()
    registry.register(record)
    path = tmp_path / "records" / f"{record.record_id}.json"
    path.write_bytes(canonical_json(record.to_dict() | {"details": {"edited": True}}))
    with pytest.raises(ValueError, match="identity"):
        registry.get(record.record_id)
    with pytest.raises(ValueError, match="identity"):
        registry.register(record)
    path.write_bytes(canonical_json(component(name="other").to_dict()))
    with pytest.raises(ValueError, match="filename"):
        registry.get(record.record_id)


def test_atomic_visibility(tmp_path, monkeypatch):
    registry = Registry(tmp_path)
    record = component()
    staged, publish = Event(), Event()
    original_link = os.link

    def pause_link(source, target):
        staged.set()
        assert publish.wait(5)
        original_link(source, target)

    monkeypatch.setattr(os, "link", pause_link)
    with ThreadPoolExecutor() as executor:
        result = executor.submit(registry.register, record)
        try:
            assert staged.wait(5)
            assert registry.find(record.kind, record.name) == []
            with pytest.raises(FileNotFoundError):
                registry.get(record.record_id)
        finally:
            publish.set()
        assert result.result(timeout=5) == record.record_id
    assert registry.get(record.record_id) == record


def test_failed_publication_leaves_no_record(tmp_path, monkeypatch):
    registry = Registry(tmp_path)

    def fail_link(source, target):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        registry.register(component())
    assert list((tmp_path / "records").iterdir()) == []


@pytest.mark.parametrize("conflict", [False, True])
def test_concurrent_registration(tmp_path, conflict):
    first = component(label="v1", label_role="release")
    second = replace(first, content_digest="0" * 64) if conflict else first
    barrier = Barrier(2)

    def register(record):
        barrier.wait(timeout=5)
        try:
            return Registry(tmp_path).register(record)
        except ValueError as exc:
            assert "Conflicting release label" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(register, (first, second)))
    assert results.count(None) == int(conflict)
    assert len({result for result in results if result is not None}) == 1
    assert len(list((tmp_path / "records").iterdir())) == 1


def test_portable_manifest(tmp_path):
    registry = Registry(tmp_path / "registry")
    leaf = component(name="shared")
    left = component(name="left", dependencies=(Dependency("input", leaf.record_id),))
    right = replace(left, name="right")
    root = component(
        dependencies=(
            Dependency("left", left.record_id),
            Dependency("right", right.record_id),
        )
    )
    for record in (leaf, left, right, root):
        registry.register(record)
    data = export_manifest(registry, root.record_id)
    manifest = load_json(data)
    for records, message in [
        ([root.to_dict()], "Missing dependency"),
        (manifest["records"] + [leaf.to_dict()], "Duplicate manifest record"),
        (manifest["records"] + [component(name="extra").to_dict()], "unreachable"),
        ([root.to_dict() | {"details": {"tampered": True}}], "identity"),
    ]:
        with pytest.raises(ValueError, match=message):
            verify_manifest(canonical_json(manifest | {"records": records}))
    portable = tmp_path / "portable"
    portable.mkdir()
    target = portable / "manifest.json"
    target.write_bytes(data)
    shutil.rmtree(registry.root)
    assert verify_manifest(target.read_bytes()) == (leaf, left, right, root)


def test_compatibility_has_no_hash_policy():
    assert CompatibilityResult().status == CompatibilityStatus.UNKNOWN
    result = CompatibilityResult(CompatibilityStatus.COMPATIBLE, "Same contract")
    assert result.status == "compatible"
    assert CompatibilityStatus.INCOMPATIBLE == "incompatible"
