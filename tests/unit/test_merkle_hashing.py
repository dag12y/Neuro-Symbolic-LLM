import hashlib
from pathlib import Path

import pytest

from versioning.hashing import canonical_json, hash_file, hash_identity, load_json
from versioning.nodes import Node

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_canonical_json_golden_vector() -> None:
    value = {
        "😀": "\x7f",
        "z": 0,
        "\ue000": "é\n",
        "a": [True, None, -(2**63), 2**63 - 1],
    }
    expected = (
        br'{"a":[true,null,-9223372036854775808,9223372036854775807],'
        br'"z":0,"\ue000":"\u00e9\n","\ud83d\ude00":"\u007f"}'
    )
    assert canonical_json(value) == expected
    assert canonical_json(dict(reversed(list(value.items())))) == expected
    assert load_json(expected) == value
    assert hash_identity(value) == (
        "64638a477b01c0f8bb61498b0e358bad55a78b81a90f0bd1892cc50aac9226e5"
    )
    assert canonical_json([0, 1]) != canonical_json([1, 0])
    assert canonical_json("é") != canonical_json("e\u0301")


def test_source_node_golden_vector() -> None:
    node = Node(
        kind="source",
        name="parser/example.py",
        payload={"byte_length": 3, "sha256": ABC_SHA256},
    )
    identity = node.to_dict()
    del identity["id"]
    expected = (
        b'{"canonicalization_version":1,"dependencies":[],"kind":"source",'
        b'"name":"parser/example.py","payload":{"byte_length":3,"sha256":"'
        + ABC_SHA256.encode("ascii")
        + b'"},"schema_version":1}'
    )
    assert canonical_json(identity) == expected
    assert node.id == "e93b978a7df4977e71d33d713d2ed159f457bb8c88e8ed0cf41e8a0e9600f6b0"


@pytest.mark.parametrize(
    "value",
    [
        -(2**63) - 1,
        2**63,
        1.0,
        {1: "text"},
        ("tuple",),
        b"bytes",
        "\ud800",
        {"\udfff": True},
    ],
)
def test_invalid_python_json_values(value: object) -> None:
    with pytest.raises(ValueError):
        canonical_json(value)


@pytest.mark.parametrize(
    "data",
    [
        b'{"outer":{"a":1,"a":2}}',
        br'{"a":1,"\u0061":2}',
        b"1.0",
        b"1e2",
        b"NaN",
        b"Infinity",
        b"-9223372036854775809",
        b"9223372036854775808",
        br'"\ud800"',
        b'"\xff"',
        '"a"'.encode("utf-16"),
    ],
)
def test_invalid_serialized_json_values(data: bytes) -> None:
    with pytest.raises(ValueError):
        load_json(data)


def test_circular_json_is_rejected() -> None:
    value: list[object] = []
    value.append(value)
    with pytest.raises(ValueError, match="Circular"):
        canonical_json(value)


def test_raw_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "raw"
    data = b"abc" * 400_000
    path.write_bytes(data)
    assert hash_file(path) == hashlib.sha256(data).hexdigest()
    path.write_bytes(b"abc")
    assert hash_file(path) == ABC_SHA256
