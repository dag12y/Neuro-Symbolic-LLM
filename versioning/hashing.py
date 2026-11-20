"""Version 1 canonical JSON and content hashing."""

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import BinaryIO

CANONICALIZATION_VERSION = 1


def _validate_string(value: str) -> None:
    # Reject surrogate code points that can alias scalar values in ASCII JSON.
    value.encode("utf-8")


def _validate_json(value: object) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            _validate_string(key)
            _validate_json(item)
    elif type(value) is list:
        for item in value:
            _validate_json(item)
    elif type(value) is str:
        _validate_string(value)
    elif type(value) not in (int, float, bool, type(None)):
        raise ValueError(f"Unsupported JSON type: {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    """Serialize a JSON value as sorted, compact ASCII JSON bytes."""
    _validate_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json(data: bytes | str) -> object:
    """Load strict JSON, including checks for duplicate keys and float overflow."""
    value = json.loads(data, object_pairs_hook=_unique_pairs)
    canonical_json(value)
    return value


def hash_identity(value: object) -> str:
    """Hash a version 1 component identity payload."""
    return hashlib.sha256(
        b"versioning:component:v1\0" + canonical_json(value)
    ).hexdigest()


def _hash_stream(stream: BinaryIO) -> str:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_file(path: str | Path) -> str:
    """Hash raw file bytes, independent of the path and timestamps."""
    with Path(path).open("rb") as stream:
        return _hash_stream(stream)


def _hash_entries(directory_fd: int, prefix: str, entries: list[list[str]]) -> None:
    for name in os.listdir(directory_fd):
        relative = prefix + name
        # Resolve children relative to the open parent without following links.
        fd = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
        )
        try:
            mode = os.fstat(fd).st_mode
            if stat.S_ISDIR(mode):
                entries.append([relative, "directory"])
                _hash_entries(fd, relative + "/", entries)
            elif stat.S_ISREG(mode):
                with open(fd, "rb", closefd=False) as stream:
                    entries.append([relative, "file", _hash_stream(stream)])
            else:
                raise ValueError(f"Unsupported directory entry: {relative!r}")
        finally:
            os.close(fd)


def hash_directory(path: str | Path) -> str:
    """Hash a POSIX directory tree, including relative names and empty directories."""
    root = Path(path)
    if ".." in root.parts:
        raise ValueError("Directory path must not contain '..'")
    entries: list[list[str]] = []
    # Walk from an open anchor so symlinks in root ancestors are rejected as well.
    fd = os.open(root.anchor or ".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for part in root.parts[1:] if root.anchor else root.parts:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        _hash_entries(fd, "", entries)
    finally:
        os.close(fd)
    entries.sort()
    return hashlib.sha256(
        b"versioning:directory:v1\0" + canonical_json(entries)
    ).hexdigest()
