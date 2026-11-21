from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from shutil import copyfile, copytree
from types import ModuleType
from typing import Any, cast

import pytest

from parser.ontology.catalog import load_manifest_catalog
from parser.semantic import model_config
from parser.semantic.backends import CallableBackend
from parser.semantic.schema import SemanticParseResult
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParserConfig,
)
from tests.semantic_fixtures import result as parse_fixture
from tests.semantic_fixtures import target
from versioning import execution
from versioning.__main__ import main
from versioning.hashing import hash_file
from versioning.manifest import export_manifest, verify_manifest
from versioning.nodes import Dependency, Node
from versioning.registry import Registry
from versioning.symbolic import (
    build_m2a_symbolic_definition,
    build_symbolic_definition,
)

_ROOT = Path(__file__).resolve().parents[2]


class _FixtureDocument:
    def __init__(self, document_id: str, corpus_id: str, marker: str) -> None:
        self.document_id = document_id
        self.corpus_id = corpus_id
        self.marker = marker

    def canonical_json(self) -> str:
        return json.dumps(vars(self), sort_keys=True, separators=(",", ":"))


class _FixtureBundle:
    def __init__(self, data: object) -> None:
        self.data = data

    @classmethod
    def model_validate(cls, data: object) -> _FixtureBundle:
        return cls(data)

    def canonical_json(self) -> str:
        return json.dumps(self.data, sort_keys=True, separators=(",", ":"))

    def validate_document(self, document: _FixtureDocument) -> None:
        assert document.document_id


class _MinerFixture:
    marker = "original"
    exporter: ModuleType

    def adapt(
        self, result: SemanticParseResult, *, skipped_rules: list[str] | None
    ) -> _FixtureDocument:
        if result.document.source_text.startswith("Skip"):
            if skipped_rules is None:
                raise ValueError("fixture unsupported rule")
            skipped_rules.extend(("first rule", "second rule"))
        return _FixtureDocument(
            result.document.document_id, result.document.corpus_id, self.marker
        )


@pytest.fixture
def miner(monkeypatch: pytest.MonkeyPatch) -> Iterator[_MinerFixture]:
    """Keep the real export loop and bundle compiler, with local contract doubles."""
    fixture = _MinerFixture()
    bundle_name = "hybrid_miner.ontology.bundle"
    adapter_name = "symbolic_pipeline.miner_adapter"
    bundle = ModuleType(bundle_name)
    monkeypatch.setattr(bundle, "OntologyBundle", _FixtureBundle, raising=False)
    adapter = ModuleType(adapter_name)
    monkeypatch.setattr(adapter, "MinerAdapterError", ValueError, raising=False)
    monkeypatch.setattr(adapter, "to_semantic_document", fixture.adapt, raising=False)
    imported = (
        "symbolic_pipeline",
        "symbolic_pipeline.document_export",
        "symbolic_pipeline.miner_bundle",
    )
    absent = [name for name in imported if name not in sys.modules]
    monkeypatch.setitem(sys.modules, bundle_name, bundle)
    monkeypatch.setitem(sys.modules, adapter_name, adapter)
    try:
        fixture.exporter = importlib.import_module("symbolic_pipeline.document_export")
        compiler = importlib.import_module("symbolic_pipeline.miner_bundle")
        monkeypatch.setattr(fixture.exporter, "to_semantic_document", fixture.adapt)
        monkeypatch.setattr(compiler, "OntologyBundle", _FixtureBundle)
        yield fixture
    finally:
        for name in absent:
            sys.modules.pop(name, None)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    catalog, _ = load_manifest_catalog()
    path = tmp_path / "input.jsonl"
    path.write_text(
        parse_fixture("Mira left.", catalog=catalog).model_dump_json() + "\n",
        encoding="utf-8",
    )
    return path


def _nodes(registry: Registry, root: Node) -> dict[str, Node]:
    return {
        node.name: node for node in verify_manifest(export_manifest(registry, root.id))
    }


def _value(node: Node) -> dict[str, Any]:
    return cast(dict[str, Any], node.payload["value"])


def _rows(registry: Registry) -> int:
    with closing(sqlite3.connect(registry.path)) as connection:
        return cast(int, connection.execute("SELECT count(*) FROM nodes").fetchone()[0])


def test_repeat_run_and_alias_free_portable_identity(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    first = execution.record_symbolic_export_run(source, tmp_path / "one", registry)
    count = _rows(registry)
    copied = tmp_path / "elsewhere.jsonl"
    copyfile(source, copied)
    second = execution.record_symbolic_export_run(copied, tmp_path / "two", registry)
    assert second == first
    assert _rows(registry) == count
    nodes = _nodes(registry, first.root)
    prefix = "symbolic/m3/adapter-export"
    invocation = nodes[f"{prefix}/inputs"]
    assert first.root.payload["purpose"] == "run"
    assert {
        edge.slot for edge in first.root.dependencies if edge.role == "uses-artifact"
    } == {"accepted", "rejected"}
    assert (
        Dependency("uses-component", "definition", nodes["symbolic/definition"].id)
        in invocation.dependencies
    )
    assert (
        Dependency(
            "uses-semantic", "effective-declaration", nodes["ontology/miner-bundle"].id
        )
        in invocation.dependencies
    )
    accepted = nodes[f"{prefix}/output/accepted"]
    materialized = tmp_path / "one" / "documents.jsonl"
    assert accepted.payload == {
        "sha256": hash_file(materialized),
        "byte_length": materialized.stat().st_size,
    }
    assert accepted.dependencies == (
        Dependency("produced-by", "inputs", invocation.id),
    )
    assert nodes[f"{prefix}/input/parser-results.jsonl"].payload["sha256"] == hash_file(
        source
    )
    manifest = export_manifest(registry, first.root.id)
    assert str(tmp_path).encode() not in manifest
    assert b"Mira left." not in manifest


def test_input_output_and_options_change_run_identity(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    first = execution.record_symbolic_export_run(source, tmp_path / "one", registry)
    with source.open("ab") as stream:
        stream.write(b"\n")
    changed_input = execution.record_symbolic_export_run(
        source, tmp_path / "two", registry
    )
    assert (tmp_path / "one/documents.jsonl").read_bytes() == (
        tmp_path / "two/documents.jsonl"
    ).read_bytes()
    miner.marker = "different output"
    changed_output = execution.record_symbolic_export_run(
        source, tmp_path / "three", registry
    )
    changed_options = execution.record_symbolic_export_run(
        source, tmp_path / "four", registry, bundle_output=True
    )
    runs = (first, changed_input, changed_output, changed_options)
    assert len({run.root.id for run in runs}) == 4
    assert (
        len({_nodes(registry, run.root)["symbolic/definition"].id for run in runs}) == 1
    )
    assert (
        _nodes(registry, changed_input.root)["symbolic/m3/adapter-export/inputs"].id
        == _nodes(registry, changed_output.root)["symbolic/m3/adapter-export/inputs"].id
    )


def test_definition_changes_run_with_same_outputs(
    miner: _MinerFixture, source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    files = {
        node.name for node in build_symbolic_definition(_ROOT) if node.kind == "source"
    }
    for name in files | {"pyproject.toml"}:
        dest = project / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        copyfile(_ROOT / name, dest)
    copytree(_ROOT / "parser/ontology", project / "parser/ontology", dirs_exist_ok=True)
    copytree(_ROOT / "configs/parser_config", project / "configs/parser_config")
    monkeypatch.setattr(execution, "_PROJECT_ROOT", project)
    registry = Registry(tmp_path / "graph.sqlite3")
    first = execution.record_symbolic_export_run(source, tmp_path / "one", registry)
    definition = _nodes(registry, first.root)["symbolic/definition"]
    with (project / "parser/semantic/semantic_parser.py").open("ab") as stream:
        stream.write(b"\n# definition change\n")
    with pytest.raises(ValueError, match="expected definition"):
        execution.record_symbolic_export_run(
            source, tmp_path / "bad", registry, expected_definition=definition.id
        )
    second = execution.record_symbolic_export_run(source, tmp_path / "two", registry)
    assert first.root.id != second.root.id
    assert definition.id != _nodes(registry, second.root)["symbolic/definition"].id
    assert (tmp_path / "one/documents.jsonl").read_bytes() == (
        tmp_path / "two/documents.jsonl"
    ).read_bytes()


def test_rejected_and_skipped_artifacts_are_recorded(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    catalog, _ = load_manifest_catalog()
    with source.open("a", encoding="utf-8") as stream:
        stream.write(parse_fixture("Skip the rule.", catalog=catalog).model_dump_json())
        stream.write("\ninvalid\n")
    registry = Registry(tmp_path / "graph.sqlite3")
    output = tmp_path / "converted"
    run = execution.record_symbolic_export_run(
        source, output, registry, bundle_output=True, skip_unsupported_rules=True
    )
    assert (run.accepted, run.rejected, run.skipped_rules) == (2, 1, 2)
    assert {path.name for path in output.iterdir()} == {
        "documents.jsonl",
        "rejected.jsonl",
        "skipped-rules.jsonl",
        "ontology-bundle.json",
    }
    nodes = _nodes(registry, run.root)
    outcome = _value(nodes["symbolic/m3/adapter-export/outcome"])
    assert outcome == {
        "status": "completed_with_rejections",
        "accepted": 2,
        "rejected": 1,
        "skipped_rules": 2,
    }
    assert _value(nodes["symbolic/m3/adapter-export/options"]) == {
        "bundle_output": True,
        "skip_unsupported_rules": True,
    }
    assert {
        edge.slot for edge in run.root.dependencies if edge.role == "uses-artifact"
    } == {"accepted", "rejected", "skipped-rules", "ontology-bundle"}


def test_empty_optional_artifact_is_present_when_requested(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    output = tmp_path / "converted"
    run = execution.record_symbolic_export_run(
        source, output, registry, skip_unsupported_rules=True
    )
    assert (output / "skipped-rules.jsonl").read_bytes() == b""
    assert (
        _nodes(registry, run.root)[
            "symbolic/m3/adapter-export/output/skipped-rules"
        ].payload["byte_length"]
        == 0
    )


def test_existing_artifacts_are_protected(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    existing = output / "documents.jsonl"
    existing.write_bytes(b"original bytes")
    registry = Registry(tmp_path / "graph.sqlite3")
    with pytest.raises(FileExistsError):
        execution.record_symbolic_export_run(source, output, registry)
    assert existing.read_bytes() == b"original bytes"
    assert not registry.path.exists()


@pytest.mark.parametrize("failure", ["missing", "corrupt", "partial", "input"])
def test_failed_export_never_registers_run(
    miner: _MinerFixture,
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    original = miner.exporter.export_semantic_documents

    def sabotaged(*args: Any, **kwargs: Any) -> Any:
        if failure == "partial":
            args[1].write_bytes(b"partial")
            raise OSError("producer stopped")
        result = original(*args, **kwargs)
        if failure == "missing":
            result.rejected_path.unlink()
        elif failure == "corrupt":
            result.output_path.write_bytes(b"not JSONL\n")
        else:
            source.write_bytes(b"input was replaced\n")
        return result

    monkeypatch.setattr(miner.exporter, "export_semantic_documents", sabotaged)
    registry = Registry(tmp_path / "graph.sqlite3")
    output = tmp_path / "converted"
    with pytest.raises((OSError, ValueError)):
        execution.record_symbolic_export_run(source, output, registry)
    assert not output.exists()
    assert not registry.path.exists()
    assert not list(tmp_path.glob(".symbolic-export-*"))


def test_published_bytes_are_rechecked_before_registration(
    miner: _MinerFixture, source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = execution.os.link

    def replace_after_link(original: Path, dest: str, *, dst_dir_fd: int) -> None:
        link(original, dest, dst_dir_fd=dst_dir_fd)
        if dest == "documents.jsonl":
            original.write_bytes(b'{"changed":true}\n')

    monkeypatch.setattr(execution.os, "link", replace_after_link)
    registry = Registry(tmp_path / "graph.sqlite3")
    with pytest.raises(ValueError, match="Published export artifact changed"):
        execution.record_symbolic_export_run(source, tmp_path / "converted", registry)
    assert not registry.path.exists()


def test_run_publication_rolls_back_entire_registry_batch(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    old = build_m2a_symbolic_definition(_ROOT)
    registry.register_batch(old)
    before = _rows(registry)
    with closing(sqlite3.connect(registry.path)) as conn, conn:
        conn.execute(
            """
            CREATE TRIGGER reject_run BEFORE INSERT ON nodes
            WHEN instr(CAST(NEW.canonical AS TEXT), '"purpose":"run"') > 0
            BEGIN SELECT RAISE(ABORT, 'fixture write failure'); END
            """
        )
    output = tmp_path / "converted"
    with pytest.raises(sqlite3.IntegrityError, match="fixture write failure"):
        execution.record_symbolic_export_run(source, output, registry)
    assert _rows(registry) == before
    assert registry.get(old[-1].id) == old[-1]
    assert (output / "documents.jsonl").is_file()


def test_parser_command_uses_real_export_with_local_model_double(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "Mira left."
    inputs = tmp_path / "source.jsonl"
    inputs.write_text(
        json.dumps({"document_id": "one", "corpus_id": "fixture", "source_text": text})
        + "\ninvalid\n",
        encoding="utf-8",
    )

    def factory(
        config: Path, *, catalog: Any, require_structured_output: bool
    ) -> ReferenceSemanticParser:
        assert config.name == "model-backend.yaml"
        assert not require_structured_output
        return ReferenceSemanticParser(
            backend=CallableBackend(lambda **_: json.dumps(target(text))),
            config=SemanticParserConfig(model_name="fixture", enable_routing=False),
            catalog=catalog,
        )

    monkeypatch.setattr(model_config, "build_reference_semantic_parser", factory)
    output = tmp_path / "parsed"
    db = tmp_path / "graph.sqlite3"
    assert (
        main(
            [
                "run-parser-export",
                "--input",
                str(inputs),
                "--output-dir",
                str(output),
                "--registry",
                str(db),
                "--student-pairs",
                "--include-metta",
            ]
        )
        == 2
    )
    root = Registry(db).get(capsys.readouterr().out.strip())
    nodes = _nodes(Registry(db), root)
    assert {path.name for path in output.iterdir()} == {
        "parser-results.jsonl",
        "rejected.jsonl",
        "metadata.jsonl",
        "student-pairs.jsonl",
    }
    parsed = SemanticParseResult.model_validate_json(
        (output / "parser-results.jsonl").read_bytes()
    )
    assert parsed.document.document_id == "one"
    assert (
        Dependency(
            "uses-semantic",
            "effective-declaration",
            nodes["configuration/parser-backend"].id,
        )
        in nodes["symbolic/m3/parser-export/inputs"].dependencies
    )
    assert _value(nodes["symbolic/m3/parser-export/outcome"]) == {
        "accepted": 1,
        "rejected": 1,
        "skipped_rules": 0,
        "status": "completed_with_rejections",
    }


def test_m3_manifest_verifies_offline_alongside_historical_definitions(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    definitions = [
        build_m2a_symbolic_definition(_ROOT),
        build_symbolic_definition(_ROOT),
    ]
    for definition in definitions:
        registry.register_batch(definition)
    archived = {d[-1].id: export_manifest(registry, d[-1].id) for d in definitions}
    output = tmp_path / "converted"
    run = execution.record_symbolic_export_run(source, output, registry)
    for root_id, manifest in archived.items():
        assert export_manifest(registry, root_id) == manifest
    manifest = export_manifest(registry, run.root.id)
    registry.path.unlink()
    for path in output.iterdir():
        path.unlink()
    assert verify_manifest(manifest, expected_root_id=run.root.id)[-1] == run.root
    # The manifest verifies after the files are gone; this is not execution proof.
    for root_id, data in archived.items():
        assert verify_manifest(data, expected_root_id=root_id)[-1].id == root_id


def test_rehashed_artifact_claim_is_not_evidence_of_execution(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    run = execution.record_symbolic_export_run(source, tmp_path / "converted", registry)
    accepted = _nodes(registry, run.root)["symbolic/m3/adapter-export/output/accepted"]
    fictional = Node(
        kind="artifact",
        name=accepted.name,
        payload={"sha256": "0" * 64, "byte_length": 10},
        dependencies=accepted.dependencies,
    )
    root = Node(
        kind="composite",
        name=run.root.name,
        payload=cast(dict[str, object], run.root.to_dict()["payload"]),
        dependencies=tuple(
            (
                Dependency(edge.role, edge.slot, fictional.id)
                if edge.target_id == accepted.id
                else edge
            )
            for edge in run.root.dependencies
        ),
    )
    registry.register_batch((fictional, root))
    assert root.id != run.root.id
    assert verify_manifest(export_manifest(registry, root.id))[-1] == root
    assert fictional.payload["sha256"] != hash_file(
        tmp_path / "converted/documents.jsonl"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "role",
        "producer",
        "definition",
        "version",
        "status",
        "count",
        "impossible-count",
        "empty-digest",
        "declaration",
        "options",
        "duplicate",
        "hidden",
    ],
)
def test_rehashed_invalid_recipe(
    miner: _MinerFixture, source: Path, tmp_path: Path, mutation: str
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    root = execution.record_symbolic_export_run(source, tmp_path / "out", registry).root
    original = verify_manifest(export_manifest(registry, root.id))
    by_name = {node.name: node for node in original}
    ids: dict[str, str] = {}
    changed = []
    for old in original:
        data = old.to_dict()
        payload = cast(dict[str, Any], data["payload"])
        deps = [Dependency(e.role, e.slot, ids[e.target_id]) for e in old.dependencies]
        name = old.name
        if name.endswith("/run"):
            if mutation == "missing":
                deps = [e for e in deps if e.slot != "accepted"]
            if mutation == "extra":
                deps.append(
                    Dependency(
                        "uses-artifact",
                        "spare",
                        next(e.target_id for e in deps if e.slot == "accepted"),
                    )
                )
            if mutation == "role":
                deps = [
                    Dependency(
                        e.role,
                        "metadata" if e.slot == "accepted" else e.slot,
                        e.target_id,
                    )
                    for e in deps
                ]
            if mutation == "version":
                payload["recipe"]["version"] = 2
        if name.endswith("/output/accepted"):
            if mutation == "producer":
                deps = [
                    Dependency("produced-by", "inputs", by_name["symbolic/parser"].id)
                ]
            if mutation == "duplicate":
                name = name.replace("accepted", "rejected")
        if mutation == "empty-digest" and name.endswith("/output/rejected"):
            payload["sha256"] = "1" * 64
        if name.endswith("/inputs"):
            if mutation == "definition":
                deps = [
                    Dependency(
                        e.role,
                        e.slot,
                        (
                            by_name["symbolic/parser"].id
                            if e.slot == "definition"
                            else e.target_id
                        ),
                    )
                    for e in deps
                ]
            if mutation == "declaration":
                deps = [
                    Dependency(
                        e.role,
                        e.slot,
                        (
                            by_name["configuration/parser-backend"].id
                            if e.slot == "effective-declaration"
                            else e.target_id
                        ),
                    )
                    for e in deps
                ]
        if name.endswith("/options"):
            if mutation == "options":
                payload["value"]["bundle_output"] = True
            if mutation == "hidden":
                spare = Node(
                    kind="source",
                    name="extra-source",
                    payload={"byte_length": 1, "sha256": "0" * 64},
                )
                changed.append(spare)
                deps.append(Dependency("uses-source", "extra", spare.id))
        if name.endswith("/outcome"):
            if mutation == "status":
                payload["value"]["status"] = "failed"
            if mutation == "impossible-count":
                payload["value"]["accepted"] = 2**60
            if mutation == "count":
                payload["value"]["accepted"] = True
        new = Node(kind=old.kind, name=name, payload=payload, dependencies=tuple(deps))
        changed.append(new)
        ids[old.id] = new.id
    registry.register_batch(changed)
    verified = verify_manifest(export_manifest(registry, ids[root.id]))
    with pytest.raises(ValueError):
        execution.validate_symbolic_run(verified, ids[root.id])


def test_explicit_artifact_verification(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    out = tmp_path / "out"
    root = execution.record_symbolic_export_run(source, out, registry).root
    nodes = verify_manifest(export_manifest(registry, root.id))
    paths = {"accepted": out / "documents.jsonl", "rejected": out / "rejected.jsonl"}
    execution.verify_run_artifacts(nodes, root.id, paths)
    alias = tmp_path / "alias"
    alias.symlink_to(paths["accepted"])
    invalid_paths: tuple[dict[str, Path], ...] = (
        {},
        {**paths, "extra": alias},
        {**paths, "accepted": alias},
        {**paths, "accepted": out},
        {**paths, "accepted": tmp_path / "absent"},
    )
    for invalid in invalid_paths:
        with pytest.raises(ValueError):
            execution.verify_run_artifacts(nodes, root.id, invalid)
    data = paths["accepted"].read_bytes()
    paths["accepted"].write_bytes(b"x" * len(data))
    with pytest.raises(ValueError, match="mismatch: accepted"):
        execution.verify_run_artifacts(nodes, root.id, paths)


def test_partial_publication_retry(
    miner: _MinerFixture, source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    out = tmp_path / "out"
    link = execution.os.link

    def interrupt(src: Path, dst: str, *, dst_dir_fd: int) -> None:
        if dst == "rejected.jsonl":
            raise OSError("link failed")
        link(src, dst, dst_dir_fd=dst_dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(execution.os, "link", interrupt)
        with pytest.raises(OSError, match="link failed"):
            execution.record_symbolic_export_run(source, out, registry)
    assert sorted(p.name for p in out.iterdir()) == ["documents.jsonl"]
    assert not registry.path.exists()
    with pytest.raises(FileExistsError):
        execution.record_symbolic_export_run(source, out, registry)
    retry = execution.record_symbolic_export_run(source, tmp_path / "retry", registry)
    assert registry.get(retry.root.id) == retry.root


def test_parser_rejects_stale_loaded_grammar(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from parser.grammar import atomese

    monkeypatch.setitem(atomese.ARITY, "stale_relation", 2)
    registry = Registry(tmp_path / "graph.sqlite3")
    with pytest.raises(ValueError, match="Loaded Atomese grammar"):
        execution.record_parser_export_run(source, tmp_path / "parsed", registry)
    assert not registry.path.exists()


def test_source_cannot_be_sqlite_sidecar(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    registry = Registry(tmp_path / "graph.sqlite3")
    sidecar = tmp_path / "graph.sqlite3-journal"
    copyfile(source, sidecar)
    with pytest.raises(ValueError, match="Registry must be separate"):
        execution.record_symbolic_export_run(sidecar, tmp_path / "converted", registry)
    assert sidecar.read_bytes() == source.read_bytes()
    assert not registry.path.exists()


def test_concurrent_exports_preserve_winner(
    miner: _MinerFixture, source: Path, tmp_path: Path
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    registry = Registry(tmp_path / "graph.sqlite3")
    output = tmp_path / "shared"

    def record() -> str | None:
        try:
            return execution.record_symbolic_export_run(
                source, output, registry
            ).root.id
        except FileExistsError:
            return None

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(record) for _ in range(2)]
        results = [future.result() for future in futures]
    assert results.count(None) == 1
    root = next(value for value in results if value is not None)
    nodes = verify_manifest(export_manifest(registry, root))
    execution.verify_run_artifacts(
        nodes,
        root,
        {"accepted": output / "documents.jsonl", "rejected": output / "rejected.jsonl"},
    )


def test_directory_substitution_cannot_redirect_links(
    miner: _MinerFixture, source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, moved, other = (tmp_path / name for name in ("out", "moved", "other"))
    other.mkdir()
    link = execution.os.link

    def substitute(src: Path, dst: str, *, dst_dir_fd: int) -> None:
        if dst == "documents.jsonl":
            output.rename(moved)
            output.symlink_to(other, target_is_directory=True)
        link(src, dst, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(execution.os, "link", substitute)
    registry = Registry(tmp_path / "graph.sqlite3")
    with pytest.raises((OSError, ValueError)):
        execution.record_symbolic_export_run(source, output, registry)
    assert list(other.iterdir()) == []
    assert {p.name for p in moved.iterdir()} == {"documents.jsonl", "rejected.jsonl"}
    assert not registry.path.exists()
