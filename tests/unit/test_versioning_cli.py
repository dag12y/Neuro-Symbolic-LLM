"""Exercise the module entry point with local registries and portable artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from versioning import (
    ComponentRecord,
    Dependency,
    Registry,
    canonical_json,
    export_manifest,
    hash_file,
)
from versioning.checkpoint import checkpoint_manifest_path, checkpoint_save
from versioning.export import export_sidecar_path, publish_export_sidecar

ROOT = Path(__file__).resolve().parents[2]


def cli(*args, stdlib_only=True):
    command = [sys.executable]
    if stdlib_only:
        command.append("-S")
    return subprocess.run(
        [*command, "-m", "versioning", *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def json_output(result, exit_code=0):
    assert result.returncode == exit_code, result.stderr or result.stdout
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "command",
    [
        None,
        "register-semantic",
        "list",
        "inspect",
        "export-manifest",
        "verify-manifest",
        "verify-export",
        "verify-checkpoint",
        "status",
    ],
)
def test_help(command):
    result = cli(*([command] if command else []), "--help")
    assert result.returncode == 0
    assert "usage: python -m versioning" in result.stdout
    if command:
        assert command in result.stdout


def test_semantic_registration_inspection_and_manifest(tmp_path):
    registry = tmp_path / "registry"
    options = ("--registry", registry)
    registration = json_output(
        cli(
            "register-semantic", "--project-root", ROOT, *options, "--json",
            stdlib_only=False,
        )
    )
    parser_id = registration["parser"]
    listed = json_output(cli("list", *options, "--json"))
    assert len(listed) == 15
    assert [item["record_id"] for item in listed] == sorted(
        item["record_id"] for item in listed
    )
    inspected = json_output(cli("inspect", parser_id, *options, "--json"))
    assert inspected["record_id"] == parser_id
    assert len(inspected["content_digest"]) == 64
    assert inspected["dependencies"]
    assert json_output(cli("status", *options, "--json"))["record_count"] == 15

    result = cli("export-manifest", parser_id, *options)
    assert json_output(result)["root_id"] == parser_id
    manifest = tmp_path / "manifest.json"
    export_args = ("export-manifest", parser_id, *options, "--output", manifest)
    assert cli(*export_args).returncode == 0
    assert cli(*export_args).returncode == 1
    assert json_output(cli("verify-manifest", manifest, "--json")) == {
        "status": "verified", "root_id": parser_id, "record_count": 15,
    }
    payload = json.loads(manifest.read_bytes())
    payload["records"][0]["name"] = "changed"
    manifest.write_bytes(canonical_json(payload))
    tampered = cli("verify-manifest", manifest)
    assert tampered.returncode == 1
    assert "identity" in tampered.stderr


def record(**overrides):
    return ComponentRecord(**{
        "kind": "prompt", "name": "demo", "content_digest": "a" * 64,
        "digest_scope": "test bytes", "label": "v1", "label_role": "release",
        "details": {"private": "do-not-print"},
        **overrides,
    })


def test_filters_and_exact_lookup(tmp_path):
    registry = Registry(tmp_path / "registry")
    expected = registry.register(record())
    registry.register(record(kind="ontology"))
    registry.register(record(name="other"))
    registry.register(record(label="v2"))
    options = ("--registry", registry.root)
    filters = ("--kind", "prompt", "--name", "demo", "--label", "v1")
    matches = json_output(cli("list", *options, *filters, "--json"))
    assert [item["record_id"] for item in matches] == [expected]
    inspected = cli("inspect", expected, *options, "--json")
    assert json_output(inspected)["record_id"] == expected
    assert "do-not-print" not in inspected.stdout
    invalid = cli("inspect", expected[:12], *options)
    assert invalid.returncode == 1
    assert "SHA-256" in invalid.stderr
    missing = cli("inspect", "0" * 64, *options)
    assert missing.returncode == 1 and "Missing" in missing.stderr


@pytest.mark.parametrize("damage", ["malformed", "release", "missing-dependency"])
def test_status_detects_corruption(tmp_path, damage):
    registry = Registry(tmp_path / "registry")
    registry.register(record())
    if damage == "malformed":
        (registry.root / "records" / ("0" * 64 + ".json")).write_text("{")
        expected = "Invalid JSON"
    else:
        if damage == "release":
            bad = record(content_digest="b" * 64)
            expected = "Conflicting release label"
        else:
            bad = record(label="v2", dependencies=(Dependency("input", "b" * 64),))
            expected = "Missing dependency"
        (registry.root / "records" / f"{bad.record_id}.json").write_bytes(
            canonical_json(bad.to_dict())
        )
    result = json_output(cli("status", "--registry", registry.root, "--json"), 1)
    assert result["status"] == "error"
    assert expected in result["message"]


def test_missing_paths_and_usage(tmp_path):
    missing = tmp_path / "missing"
    for args in (
        ("list", "--registry", missing),
        ("status", "--registry", missing),
        ("verify-manifest", missing),
        ("verify-export", missing),
        ("verify-checkpoint", missing),
    ):
        result = cli(*args)
        assert result.returncode == 1
        assert result.stderr.startswith("Error:")
    assert cli("inspect").returncode == 2


def test_export_verification(tmp_path):
    registry = Registry(tmp_path / "registry")
    parser_id = registry.register(record(kind="parser-implementation"))
    output = tmp_path / "results.jsonl"
    output.write_bytes(b"{}\n")
    digest = hash_file(output)
    publish_export_sidecar(
        output,
        expected_sha256=digest,
        source_sha256=digest,
        parser_record_id=parser_id,
        component_manifest=export_manifest(registry, parser_id),
        settings={},
        unresolved=[],
    )
    for path in (output, export_sidecar_path(output)):
        verified = json_output(cli("verify-export", path, "--json"))
        assert verified["content_digest"] == digest
    output.write_bytes(b"tampered")
    assert "SHA-256 mismatch" in cli("verify-export", output).stderr
    output.unlink()
    missing = cli("verify-export", export_sidecar_path(output))
    assert missing.returncode == 1
    assert "Export artifact not found" in missing.stderr


def test_checkpoint_verification(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    legacy = json_output(cli("verify-checkpoint", checkpoint, "--json"), 3)
    assert legacy["status"] == "legacy"
    with pytest.raises(OSError, match="interruption"):
        with checkpoint_save(checkpoint, None):
            raise OSError("interruption")
    incomplete = cli("verify-checkpoint", checkpoint)
    assert incomplete.returncode == 1
    assert "Incomplete checkpoint" in incomplete.stderr
    sidecar = checkpoint_manifest_path(checkpoint)
    sidecar.unlink()
    missing = cli("verify-checkpoint", checkpoint)
    assert missing.returncode == 1
    assert "Tracked checkpoint manifest is missing" in missing.stderr
    with checkpoint_save(checkpoint, None):
        (checkpoint / "model.bin").write_bytes(b"checkpoint bytes")
    verified = json_output(cli("verify-checkpoint", checkpoint, "--json"))
    assert verified["status"] == "verified"
    (checkpoint / "model.bin").write_bytes(b"tampered")
    mismatch = cli("verify-checkpoint", checkpoint)
    assert mismatch.returncode == 1
    assert "SHA-256 mismatch" in mismatch.stderr
    sidecar.write_text("{")
    assert cli("verify-checkpoint", checkpoint).returncode == 1


def test_errors_omit_private_values_and_path_names(tmp_path):
    private = "private-marker"
    malformed = tmp_path / private
    # Duplicate-key diagnostics from the JSON loader contain the actual key.
    malformed.write_text('{"private-marker": 1, "private-marker": 2}')
    result = json_output(cli("verify-manifest", malformed, "--json"), 1)
    assert result["message"] == "Duplicate JSON key"
    registry = Registry(tmp_path / "registry")
    registry.register(record(label=private))
    conflicting = record(label=private, content_digest="b" * 64)
    (registry.root / "records" / f"{conflicting.record_id}.json").write_bytes(
        canonical_json(conflicting.to_dict())
    )
    status = cli("status", "--registry", registry.root)
    assert status.returncode == 1
    assert "Conflicting release label" in status.stderr
    assert private not in status.stderr
    usage = cli("list", "--unknown-option", private)
    assert usage.returncode == 2
    assert private not in usage.stderr
