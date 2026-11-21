"""Canonical JSON and SHA-256 for Merkle node identities."""

import hashlib
import json
from pathlib import Path
from typing import Never

CANONICALIZATION_VERSION = 1
NODE_DOMAIN = b"nsllm:merkle-node:v1\0"
_INT_MIN = -(2**63)
_INT_MAX = 2**63 - 1


def _validate_string(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("JSON strings must not contain lone surrogates") from error


def _validate_json(value: object, active: set[int]) -> None:
    if type(value) is dict or type(value) is list:
        identity = id(value)
        if identity in active:
            raise ValueError("Circular JSON value")
        active.add(identity)
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                _validate_string(key)
                _validate_json(item, active)
        else:
            for item in value:
                _validate_json(item, active)
        active.remove(identity)
    elif type(value) is str:
        _validate_string(value)
    elif type(value) is int:
        if not _INT_MIN <= value <= _INT_MAX:
            raise ValueError("JSON integers must fit in signed 64 bits")
    elif type(value) not in (bool, type(None)):
        raise ValueError(f"Unsupported JSON type: {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    """Encode strict JSON as compact, ASCII-escaped UTF-8 with sorted keys."""
    try:
        _validate_json(value, set())
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except RecursionError as error:
        raise ValueError("JSON value is nested too deeply") from error


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_number(_: str) -> Never:
    raise ValueError("JSON numbers must be signed 64-bit integers")


def load_json(data: bytes | str) -> object:
    """Decode strict UTF-8 JSON without losing duplicate object keys."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_pairs,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
        _validate_json(value, set())
    except RecursionError as error:
        raise ValueError("JSON value is nested too deeply") from error
    return value


def hash_identity(value: object) -> str:
    """Hash the canonical identity fields of a Merkle node."""
    return hashlib.sha256(NODE_DOMAIN + canonical_json(value)).hexdigest()


def hash_file(path: str | Path) -> str:
    """Stream a raw SHA-256 digest over the file's exact bytes."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
