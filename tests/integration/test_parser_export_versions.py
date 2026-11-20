"""Offline coverage of parser export version manifests without the miner extra."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic import catalog_document_cli
from parser.semantic import catalog_document_export as exporter
from parser.semantic.backends import CallableBackend
from parser.semantic.schema import SemanticParseResult
from parser.semantic.semantic_parser import (
    ModelGenerationError,
    ReferenceSemanticParser,
    SemanticParserConfig,
)
from tests.semantic_fixtures import target
from versioning import Dependency, canonical_json, hash_file, load_json, verify_manifest
from versioning.export import export_sidecar_path, verify_export_sidecar


@pytest.fixture
def export_case(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Mira left.\nSol sang.\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    return {
        "input_path": source,
        "output_path": output,
        "rejected_path": tmp_path / "rejected.jsonl",
    }


def _parser(calls, *, catalog=None, fail=None):
    def generate(*, prompt, model):
        text = prompt.rsplit("Source text:\n", 1)[1]
        calls.append(text)
        if text == fail:
            raise ConnectionError("synthetic backend interruption")
        return json.dumps(target(text))

    return ReferenceSemanticParser(
        backend=CallableBackend(generate, provider_name="private-provider"),
        config=SemanticParserConfig(model_name="private-model", enable_routing=False),
        catalog=catalog,
    )


def _checkpoint(output):
    return output.with_name(output.name + ".checkpoint.json")


def test_success_writes_portable_sidecar_after_checkpoint_cleanup(export_case):
    result = exporter.export_catalog_documents(_parser([]), **export_case)
    output = result.output_path
    sidecar = result.versions_path
    assert result.accepted == 2
    assert sidecar == export_sidecar_path(output)
    assert not _checkpoint(output).exists()

    payload = load_json(sidecar.read_bytes())
    records = verify_manifest(canonical_json(payload["components"]))
    artifact = verify_export_sidecar(sidecar)
    assert len(records) == 15
    assert artifact.content_digest == hash_file(output)
    assert artifact.details["source_sha256"] == hash_file(export_case["input_path"])
    assert artifact.dependencies == (
        Dependency("parser-implementation", records[-1].record_id),
    )
    assert artifact.details["settings"]["routing_enabled"] is False
    for sensitive in (b"Mira left.", b"private-provider", b"private-model"):
        assert sensitive not in sidecar.read_bytes()

    row = json.loads(output.read_text().splitlines()[0])
    assert set(row) == {"document", "provenance", "contract_version", "extraction"}
    SemanticParseResult.model_validate(row)
    portable = output.parent / "portable"
    portable.mkdir()
    output.rename(portable / output.name)
    sidecar.rename(portable / sidecar.name)
    assert verify_export_sidecar(portable / sidecar.name) == artifact


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    [
        ("output", ValueError, "artifact SHA-256 mismatch"),
        ("missing-artifact", FileNotFoundError, "artifact not found"),
        ("component", ValueError, "record fields or identity"),
        ("dependency", ValueError, "does not link"),
        ("locator", ValueError, "artifact locator"),
        ("malformed", ValueError, "Invalid export version sidecar"),
    ],
)
def test_verification_detects_tampering(export_case, mutation, error_type, message):
    result = exporter.export_catalog_documents(_parser([]), **export_case)
    output = result.output_path
    sidecar = result.versions_path
    if mutation == "output":
        output.write_bytes(output.read_bytes() + b"\n")
    elif mutation == "missing-artifact":
        output.unlink()
    elif mutation == "malformed":
        sidecar.write_bytes(b'{"artifact":')
    else:
        payload = load_json(sidecar.read_bytes())
        if mutation == "component":
            payload["components"]["records"][0]["details"]["tampered"] = True
        else:
            artifact = verify_export_sidecar(sidecar)
            if mutation == "dependency":
                changed = replace(
                    artifact,
                    dependencies=(Dependency("parser-implementation", "0" * 64),),
                )
            else:
                changed = replace(artifact, name="../output.jsonl")
            payload["artifact"] = changed.to_dict()
        sidecar.write_bytes(canonical_json(payload))
    with pytest.raises(error_type, match=message):
        verify_export_sidecar(sidecar)


def test_output_rename_failure_keeps_checkpoint_and_no_sidecar(export_case, monkeypatch):
    output = export_case["output_path"]
    original = Path.rename

    def fail_output(path, destination):
        if destination == output:
            raise OSError("synthetic rename failure")
        return original(path, destination)

    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_output)
        with pytest.raises(OSError, match="synthetic rename failure"):
            exporter.export_catalog_documents(_parser(calls), **export_case)
    assert len(calls) == 2
    assert not output.exists()
    assert not export_sidecar_path(output).exists()
    assert _checkpoint(output).exists()
    result = exporter.export_catalog_documents(_parser(calls), resume=True, **export_case)
    assert len(calls) == 2
    verify_export_sidecar(result.versions_path)


def test_sidecar_write_failure_surfaces_and_resumes_without_inference(
    export_case, monkeypatch
):
    output = export_case["output_path"]
    sidecar = export_sidecar_path(output)
    original = Path.replace
    calls = []

    def fail_sidecar(path, destination):
        if destination == sidecar:
            raise PermissionError("synthetic sidecar error")
        return original(path, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_sidecar)
        with pytest.raises(OSError, match="artifacts finalized.*version sidecar"):
            exporter.export_catalog_documents(_parser(calls), **export_case)
    assert output.exists()
    assert not sidecar.exists()
    assert not list(output.parent.glob("*.tmp"))
    assert json.loads(_checkpoint(output).read_text())["complete"]
    result = exporter.export_catalog_documents(_parser(calls), resume=True, **export_case)
    assert len(calls) == 2
    assert not _checkpoint(output).exists()
    verify_export_sidecar(result.versions_path)


def test_new_run_refuses_stale_sidecar_and_existing_output(export_case):
    output = export_case["output_path"]
    sidecar = export_sidecar_path(output)
    sidecar.write_bytes(b"stale")
    calls = []
    with pytest.raises(FileExistsError, match="versions.json"):
        exporter.export_catalog_documents(_parser(calls), **export_case)
    assert sidecar.read_bytes() == b"stale"
    assert not calls
    sidecar.unlink()
    exporter.export_catalog_documents(_parser(calls), **export_case)
    before = sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export_catalog_documents(_parser(calls), **export_case)
    assert sidecar.read_bytes() == before
    assert len(calls) == 2


def test_completed_resume_accepts_only_identical_published_sidecar(export_case, monkeypatch):
    output = export_case["output_path"]
    checkpoint = _checkpoint(output)
    original = Path.unlink
    calls = []

    def fail_cleanup(path, *args, **kwargs):
        if path == checkpoint:
            raise OSError("synthetic checkpoint cleanup failure")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_cleanup)
        with pytest.raises(OSError, match="cleanup failure"):
            exporter.export_catalog_documents(_parser(calls), **export_case)
    sidecar = export_sidecar_path(output)
    data = sidecar.read_bytes()
    sidecar.write_bytes(b"stale")
    with pytest.raises(OSError, match="Conflicting version sidecar"):
        exporter.export_catalog_documents(_parser(calls), resume=True, **export_case)
    assert sidecar.read_bytes() == b"stale"
    assert checkpoint.exists()
    sidecar.write_bytes(data)
    result = exporter.export_catalog_documents(_parser(calls), resume=True, **export_case)
    assert len(calls) == 2
    assert not checkpoint.exists()
    verify_export_sidecar(result.versions_path)


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_preserves_work_and_marks_legacy_prefix(export_case, legacy):
    output = export_case["output_path"]
    calls = []
    with pytest.raises(ModelGenerationError):
        exporter.export_catalog_documents(_parser(calls, fail="Sol sang."), **export_case)
    assert not export_sidecar_path(output).exists()
    if legacy:
        path = _checkpoint(output)
        checkpoint = json.loads(path.read_text())
        del checkpoint["versioning"]
        path.write_text(json.dumps(checkpoint))
    result = exporter.export_catalog_documents(_parser(calls), resume=True, **export_case)
    assert calls == ["Mira left.", "Sol sang.", "Sol sang."]
    artifact = verify_export_sidecar(result.versions_path)
    prefix_unresolved = any(
        "legacy checkpoint prefix" in item for item in artifact.details["unresolved"]
    )
    assert prefix_unresolved == legacy


def test_custom_catalog_records_effective_definitions(export_case):
    catalog = load_ontology_catalog(modules=("upper",))
    result = exporter.export_catalog_documents(_parser([], catalog=catalog), **export_case)
    manifest = load_json(result.versions_path.read_bytes())
    records = verify_manifest(canonical_json(manifest["components"]))
    active = [record for record in records if record.kind == "ontology-catalog"]
    assert len(active) == 1
    assert active[0].details["ontology_versions"] == catalog.versions
    assert not any(record.kind == "ontology-bundle" for record in records)
    verify_export_sidecar(result.versions_path)


def test_resume_rejects_different_component_snapshot(export_case):
    calls = []
    with pytest.raises(ModelGenerationError):
        exporter.export_catalog_documents(_parser(calls, fail="Sol sang."), **export_case)
    catalog = load_ontology_catalog()
    relations = dict(catalog.relations)
    key = next(
        name for name, relation in relations.items() if relation.ontology == "discourse"
    )
    relations[key] = replace(
        relations[key], extraction_instruction="Changed discourse guidance"
    )
    changed = replace(catalog, relations=relations)
    with pytest.raises(ValueError, match="checkpoint.*versioned components"):
        exporter.export_catalog_documents(
            _parser(calls, catalog=changed), resume=True, **export_case
        )
    assert calls == ["Mira left.", "Sol sang."]


def test_detected_source_change_invalidates_checkpoint(export_case, monkeypatch):
    source = export_case["input_path"]
    output = export_case["output_path"]
    original = source.read_bytes()
    parser = _parser([])
    parse = parser.parse_document

    def change_source(document):
        source.write_bytes(original + b"Another line.\n")
        return parse(document)

    with monkeypatch.context() as patch:
        patch.setattr(parser, "parse_document", change_source)
        with pytest.raises(ValueError, match="source changed during processing"):
            exporter.export_catalog_documents(parser, **export_case)
    source.write_bytes(original)
    assert not output.exists()
    assert not export_sidecar_path(output).exists()
    with pytest.raises(ValueError, match="checkpoint was invalidated"):
        exporter.export_catalog_documents(parser, resume=True, **export_case)


def test_cli_surfaces_sidecar_failure_and_reports_resume_success(
    export_case, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        catalog_document_cli,
        "build_reference_semantic_parser",
        lambda *args, **kwargs: _parser(calls, catalog=kwargs["catalog"]),
    )
    arguments = [
        "--input",
        str(export_case["input_path"]),
        "--output",
        str(export_case["output_path"]),
    ]

    def fail_publication(*args, **kwargs):
        raise PermissionError("synthetic publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(exporter, "publish_export_sidecar", fail_publication)
        assert catalog_document_cli.main(arguments) == 1
    failed = capsys.readouterr()
    assert "version sidecar could not be published" in failed.err
    assert "Version manifest:" not in failed.out
    assert catalog_document_cli.main([*arguments, "--resume"]) == 0
    assert "Version manifest:" in capsys.readouterr().out
    assert len(calls) == 2


def test_sidecar_records_boolean_effect_of_free_form_flag(export_case):
    parser = _parser([])
    parser._require_structured_output = "private-option-value"
    # An empty input exercises settings capture without requiring structured inference.
    export_case["input_path"].write_bytes(b"")
    result = exporter.export_catalog_documents(parser, **export_case)
    artifact = verify_export_sidecar(result.versions_path)
    assert artifact.details["settings"]["structured_output_required"] is True
    assert b"private-option-value" not in result.versions_path.read_bytes()
