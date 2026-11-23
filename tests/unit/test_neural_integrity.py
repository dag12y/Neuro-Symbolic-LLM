from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from versioning.__main__ import main
from versioning.neural_integrity import (
    dataset_observation_node,
    dataset_selector_node,
    imported_checkpoint_node,
    inspect_a1_checkpoint,
    ordered_text_rows_node,
    resolve_checkpoint_path,
    token_blocks_node,
    verify_a1_checkpoint,
)


def test_text_rows_preserve_order_boundaries_and_empty_rows() -> None:
    assert ordered_text_rows_node("data/rows", ["ab", "c"]).id != (
        ordered_text_rows_node("data/rows", ["a", "bc"]).id
    )
    assert ordered_text_rows_node("data/rows", ["a", "b"]).id != (
        ordered_text_rows_node("data/rows", ["b", "a"]).id
    )
    assert ordered_text_rows_node("data/rows", []).id != (
        ordered_text_rows_node("data/rows", [""]).id
    )


def test_dataset_selector_rows_and_prepared_blocks_are_distinct() -> None:
    selector = dataset_selector_node(
        "data/selector",
        {
            "dataset": "org/data",
            "dataset_config": None,
            "text_field": "text",
            "train_split": "train",
            "eval_split": "validation",
            "test_split": "test",
            "separator": "\\n\\n",
            "tokenizer": "org/model",
            "seq_len": 2,
            "preprocessing": "join-tokenize-drop-tail-int32-reshape/v1",
        },
    )
    rows = ordered_text_rows_node("data/rows", ["one", "two"])
    blocks = token_blocks_node(
        "data/blocks", np.array([[1, 2], [3, 4]], dtype=np.int32)
    )
    graph = dataset_observation_node("data/observation", selector, rows, blocks)
    assert graph.kind == "composite"
    assert {edge.slot for edge in graph.dependencies} == {
        "selector",
        "text-rows",
        "token-blocks",
    }
    assert len({selector.id, rows.id, blocks.id}) == 3


def test_dataset_observation_rejects_wrong_token_block_width() -> None:
    selector = dataset_selector_node(
        "data/selector",
        {
            "dataset": "org/data",
            "dataset_config": None,
            "text_field": "text",
            "train_split": "train",
            "eval_split": "validation",
            "test_split": "test",
            "separator": "\\n",
            "tokenizer": "org/model",
            "seq_len": 3,
            "preprocessing": "join-tokenize-drop-tail-int32-reshape/v1",
        },
    )
    rows = ordered_text_rows_node("data/rows", [])
    blocks = token_blocks_node("data/blocks", np.empty((0, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="width"):
        dataset_observation_node("data/observation", selector, rows, blocks)


def test_token_array_canonicalizes_endian_and_layout_but_tracks_shape() -> None:
    little = np.array([[1, 2], [3, 4]], dtype="<i4")
    big = np.array([[1, 2], [3, 4]], dtype=">i4")
    noncontiguous = np.array([[1, 3], [2, 4]], dtype=np.int32).T
    assert (
        token_blocks_node("data/blocks", little).id
        == token_blocks_node("data/blocks", big).id
    )
    assert (
        token_blocks_node("data/blocks", little).id
        == token_blocks_node("data/blocks", noncontiguous).id
    )
    assert (
        token_blocks_node("data/blocks", little).id
        != token_blocks_node("data/blocks", little.reshape(1, 4)).id
    )


@pytest.mark.parametrize(
    "array",
    [
        np.array([1, 2]),
        np.array([[1.0]], dtype=np.float32),
        np.array([[1]], dtype=np.uint32),
    ],
)
def test_token_array_rejects_unsupported_shape_or_dtype(array: np.ndarray) -> None:
    with pytest.raises(ValueError):
        token_blocks_node("data/blocks", array)


def test_empty_token_blocks_are_valid_with_zero_length_identity() -> None:
    node = token_blocks_node("data/blocks", np.empty((0, 8), dtype=np.int32))
    payload = cast(dict[str, Any], node.to_dict()["payload"])
    value = cast(dict[str, Any], payload["value"])
    assert value["shape"] == [0, 8]
    assert value["byte_length"] == 0


def _write_archive(path: Path, *, full: bool = True, value: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if full:
        np.savez(
            path,
            **{
                "params/layer_0/A": np.array([[value]], dtype=np.float32),
                "params/layer_0/B": np.array([[2.0]], dtype=np.float32),
                "opt/leaf_0": np.array([3.0], dtype=np.float32),
                "__opt_treedef__": np.array([1, 2, 3], dtype=np.uint8),
                "__metadata__": np.array(json.dumps({"step": 4})),
            },
        )
    else:
        np.savez(
            path,
            **{
                "layer_0/A": np.array([[value]], dtype=np.float32),
                "layer_0/B": np.array([[2.0]], dtype=np.float32),
            },
        )


def test_checkpoint_full_weights_only_integrity_and_no_producer_claim(
    tmp_path: Path,
) -> None:
    full = tmp_path / "checkpoints" / "step.npz"
    weights = tmp_path / "weights.npz"
    _write_archive(full)
    _write_archive(weights, full=False)
    full_info = inspect_a1_checkpoint(full)
    weights_info = inspect_a1_checkpoint(weights)
    assert full_info.classification == "full_a1"
    assert full_info.optimizer_members == ("__opt_treedef__", "opt/leaf_0")
    assert full_info.metadata_fields == ("step",)
    assert weights_info.classification == "weights_only_a1"
    assert full_info.sha256 == hashlib.sha256(full.read_bytes()).hexdigest()
    assert full_info.provenance == "unknown"
    assert imported_checkpoint_node("imported/a1", full_info).kind == "source"
    assert all(
        edge.role != "produced-by"
        for edge in imported_checkpoint_node("imported/a1", full_info).dependencies
    )


def test_checkpoint_same_shapes_different_values_have_different_bytes(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "one.npz", tmp_path / "two.npz"
    _write_archive(first, value=1.0)
    _write_archive(second, value=9.0)
    a, b = inspect_a1_checkpoint(first), inspect_a1_checkpoint(second)
    assert a.parameters[0]["shape"] == b.parameters[0]["shape"]
    assert a.sha256 != b.sha256
    assert a.parameters[0]["data_sha256"] != b.parameters[0]["data_sha256"]


def test_duplicate_zip_members_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.npz"
    npy = tmp_path / "one.npy"
    np.save(npy, np.array([1], dtype=np.int32))
    data = npy.read_bytes()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("layer_0/A.npy", data)
        archive.writestr("layer_0/A.npy", data)
    with pytest.raises(ValueError, match="duplicate ZIP"):
        inspect_a1_checkpoint(path)


def test_malformed_and_truncated_archive_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    path.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(ValueError):
        inspect_a1_checkpoint(path)


def test_malformed_npy_member_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-header.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("layer_0/A.npy", b"not an npy header")
    with pytest.raises(ValueError, match="Malformed or unsafe"):
        inspect_a1_checkpoint(path)


def test_valid_unknown_archive_and_object_array_policy(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.npz"
    np.savez(unknown, unrelated=np.array([1, 2], dtype=np.int32))
    assert inspect_a1_checkpoint(unknown).classification == "unknown"
    objects = tmp_path / "objects.npz"
    np.savez(objects, unsafe=np.array([{"pickle": True}], dtype=object))
    with pytest.raises(ValueError, match="pickle"):
        inspect_a1_checkpoint(objects)


def test_checkpoint_fallback_matches_runtime_and_reports_observed_file(
    tmp_path: Path,
) -> None:
    full = tmp_path / "run" / "checkpoints" / "step.npz"
    _write_archive(full)
    missing_params = tmp_path / "run" / "params" / "step.npz"
    assert resolve_checkpoint_path(missing_params) == full
    assert (
        inspect_a1_checkpoint(missing_params).sha256
        == hashlib.sha256(full.read_bytes()).hexdigest()
    )


def test_interrupted_checkpoint_pair_falls_back_to_weights(tmp_path: Path) -> None:
    params = tmp_path / "run" / "params" / "step.npz"
    _write_archive(params, full=False)
    missing_full = tmp_path / "run" / "checkpoints" / "step.npz"
    assert resolve_checkpoint_path(missing_full) == params
    assert inspect_a1_checkpoint(missing_full).classification == "weights_only_a1"


def test_declared_member_and_archive_limits_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large-member.npz"
    np.savez(path, data=np.array([1], dtype=np.int32))
    monkeypatch.setattr("versioning.neural_integrity.MAX_MEMBER_BYTES", 1)
    with pytest.raises(ValueError, match="member exceeds"):
        inspect_a1_checkpoint(path)


def test_central_directory_limit_is_enforced_before_member_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "central.npz"
    _write_archive(path)
    monkeypatch.setattr("versioning.neural_integrity.MAX_CENTRAL_DIRECTORY_BYTES", 1)
    with pytest.raises(ValueError, match="central directory"):
        inspect_a1_checkpoint(path)


def test_file_mutation_during_inspection_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mutate.npz"
    _write_archive(path)
    original = __import__(
        "versioning.neural_integrity", fromlist=["_inspect_zip"]
    )._inspect_zip

    def mutate(stream, size):
        result = original(stream, size)
        with path.open("ab") as changed:
            changed.write(b"x")
        return result

    monkeypatch.setattr("versioning.neural_integrity._inspect_zip", mutate)
    with pytest.raises(ValueError, match="changed"):
        inspect_a1_checkpoint(path)


def test_checkpoint_exact_digest_and_length_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "weights.npz"
    _write_archive(path, full=False)
    observed = inspect_a1_checkpoint(path)
    verified = verify_a1_checkpoint(
        path, expected_digest=observed.sha256, expected_length=observed.byte_length
    )
    assert verified.sha256 == observed.sha256
    assert verified.byte_length == observed.byte_length
    assert (
        main(
            [
                "verify-a1-checkpoint",
                str(path),
                "--expected-digest",
                observed.sha256,
                "--expected-length",
                str(observed.byte_length),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["exact_byte_integrity"] == "verified"
    assert output["structural_inspection"] == "not_performed"
    assert output["authorship"] == "unknown"
    with pytest.raises(ValueError, match="mismatch"):
        verify_a1_checkpoint(path, expected_digest="0" * 64)


def test_exact_byte_verification_is_independent_of_archive_structure(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.npz"
    malformed.write_bytes(b"not a zip archive")
    digest = hashlib.sha256(malformed.read_bytes()).hexdigest()
    assert verify_a1_checkpoint(malformed, expected_digest=digest).sha256 == digest
    with pytest.raises(ValueError):
        inspect_a1_checkpoint(malformed)


def test_fallback_prefers_existing_requested_path_over_companion(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "checkpoints" / "step.npz"
    companion = tmp_path / "params" / "step.npz"
    _write_archive(requested)
    _write_archive(companion, full=False)
    assert resolve_checkpoint_path(requested) == requested
