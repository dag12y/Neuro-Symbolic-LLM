"""Student checkpoint lifecycle with local, deterministic model/tokenizer doubles."""

import json

import pytest
import torch
import yaml

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_student import validate_catalog_student_checkpoint
from parser.student import train as training
from tests.semantic_fixtures import pair, write_checkpoint_metadata
from tests.unit.test_student_train import TinyModel
from versioning import canonical_json, hash_directory, hash_file, verify_manifest
from versioning.checkpoint import checkpoint_manifest_path, verify_student_checkpoint


class SavedModel(TinyModel):
    def save_pretrained(self, directory):
        (directory / "model-00001-of-00002.safetensors").write_bytes(b"first")
        (directory / "model-00002-of-00002.safetensors").write_bytes(b"second")
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                "first": "model-00001-of-00002.safetensors",
                "second": "model-00002-of-00002.safetensors",
            }}),
            encoding="utf-8",
        )


class Tokenizer:
    pad_token = "pad"
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2]}

    def save_pretrained(self, directory):
        (directory / "tokenizer.json").write_text("{}")


@pytest.mark.parametrize(
    ("model_id", "expected_id"),
    [("example/tiny-model", "example/tiny-model"), ("private-bare-locator", None)],
)
def test_training_publishes_portable_checkpoint(tmp_path, monkeypatch, model_id, expected_id):
    model = SavedModel()
    model.config._commit_hash = "a" * 40
    monkeypatch.setattr(training.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    config = training.load_config()
    config["training"].update(epochs=1, batch_size=1, bf16=False)
    config["student"]["base_model"] = model_id
    config["private_option"] = "private-config-value"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    train_file = tmp_path / "private-train.jsonl"
    validation_file = tmp_path / "private-val.jsonl"
    train_file.write_text(json.dumps(pair("Private training sentence.")) + "\n")
    validation_file.write_text(json.dumps(pair("Private validation sentence.")) + "\n")
    directory = tmp_path / "student"
    training.train(str(train_file), str(config_path), str(directory), val_file=str(validation_file))

    checkpoint = verify_student_checkpoint(directory)
    assert checkpoint.content_digest == hash_directory(directory)
    path = checkpoint_manifest_path(directory)
    raw = path.read_bytes()
    for private in (b"private-config-value", b"Private training sentence", b"private-train", str(tmp_path).encode()):
        assert private not in raw
    records = verify_manifest(canonical_json(json.loads(raw)["components"]))
    assert records[-1] == checkpoint
    provenance = next(record for record in records if record.kind == "student-training")
    assert checkpoint.dependencies[0].record_id == provenance.record_id
    assert {dep.role for dep in provenance.dependencies} == {"student-prompt", "ontology"}
    assert provenance.details["train_file_sha256"] == hash_file(train_file)
    assert provenance.details["validation_file_sha256"] == hash_file(validation_file)
    assert provenance.details["datasets"]["training"]["ordered_token_examples_sha256"]
    settings = provenance.details["settings"]
    assert settings["seed"] == config["training"]["seed"]
    assert settings["base_model_revision"] == "a" * 40
    assert settings["base_model"] == expected_id
    if expected_id is None:
        assert model_id.encode() not in raw
    validate_catalog_student_checkpoint(directory, catalog=load_ontology_catalog())

    moved = tmp_path / "moved"
    directory.rename(moved)
    path.rename(checkpoint_manifest_path(moved))
    assert verify_student_checkpoint(moved) == checkpoint
    shard = moved / "model-00002-of-00002.safetensors"
    shard.unlink()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_student_checkpoint(moved)
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_student_checkpoint(moved)
    shard.write_bytes(b"second")
    value = json.loads(raw)
    value["components"]["records"][0]["content_digest"] = "0" * 64
    checkpoint_manifest_path(moved).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        verify_student_checkpoint(moved)


def test_interrupted_overwrite_and_retry(tmp_path):
    directory = tmp_path / "student"
    training.save_model(SavedModel(), Tokenizer(), str(directory), {})
    original = verify_student_checkpoint(directory)
    assert original.details["training_provenance"] == "unavailable"

    class Interrupted(Tokenizer):
        def save_pretrained(self, directory):
            raise OSError("simulated tokenizer save failure")

    with pytest.raises(OSError, match="simulated tokenizer"):
        training.save_model(SavedModel(), Interrupted(), str(directory), {})
    assert not (directory / "training_metadata.json").exists()
    with pytest.raises(ValueError, match="Incomplete checkpoint save"):
        verify_student_checkpoint(directory)
    training.save_model(SavedModel(), Tokenizer(), str(directory), {})
    assert verify_student_checkpoint(directory) == original


def test_legacy_and_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_student_checkpoint(tmp_path / "missing")
    write_checkpoint_metadata(tmp_path)
    assert verify_student_checkpoint(tmp_path) is None
    validate_catalog_student_checkpoint(tmp_path, catalog=load_ontology_catalog())


def test_stale_manifest_refuses_overwrite(tmp_path):
    directory = tmp_path / "model"
    training.save_model(SavedModel(), Tokenizer(), str(directory), {})
    sidecar = checkpoint_manifest_path(directory)
    prior = sidecar.read_bytes()
    shard = directory / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"externally replaced")
    with pytest.raises(ValueError, match="Stale checkpoint manifest"):
        training.save_model(SavedModel(), Tokenizer(), str(directory), {})
    assert sidecar.read_bytes() == prior
    assert shard.read_bytes() == b"externally replaced"


def test_manifest_publication_failure_leaves_incomplete_marker(tmp_path, monkeypatch):
    from versioning import checkpoint as tracking

    directory = tmp_path / "model"
    write = tracking._write

    def fail_manifest(path, data):
        if path == checkpoint_manifest_path(directory) and "components" in json.loads(data):
            assert (directory / "training_metadata.json").is_file()
            raise OSError("simulated manifest publication failure")
        write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(tracking, "_write", fail_manifest)
        with pytest.raises(OSError, match="publication failure"):
            training.save_model(SavedModel(), Tokenizer(), str(directory), {})
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        verify_student_checkpoint(directory)
    training.save_model(SavedModel(), Tokenizer(), str(directory), {})
    assert verify_student_checkpoint(directory) is not None


def test_interrupted_checkpoint_without_sidecar_is_not_legacy(tmp_path):
    class Interrupted(Tokenizer):
        def save_pretrained(self, directory):
            raise OSError("interrupted save")

    directory = tmp_path / "student"
    with pytest.raises(OSError, match="interrupted"):
        training.save_model(SavedModel(), Interrupted(), str(directory), {})
    checkpoint_manifest_path(directory).unlink()
    with pytest.raises(ValueError, match="Tracked checkpoint manifest is missing"):
        verify_student_checkpoint(directory)


def test_overwrite_retires_previous_weight_layout(tmp_path):
    class Unsharded(SavedModel):
        def save_pretrained(self, directory):
            (directory / "model.safetensors").write_bytes(b"unsharded")

    directory = tmp_path / "student"
    directory.mkdir()
    notes = directory / "notes.txt"
    notes.write_bytes(b"keep")
    training.save_model(Unsharded(), Tokenizer(), str(directory), {})
    training.save_model(SavedModel(), Tokenizer(), str(directory), {})
    assert not (directory / "model.safetensors").exists()
    assert (directory / "model.safetensors.index.json").is_file()
    assert notes.read_bytes() == b"keep"
    assert verify_student_checkpoint(directory).content_digest == hash_directory(directory)
