"""Offline identities for observed Stage A1 data and checkpoint bytes.

These observations describe supplied bytes and arrays. They do not establish
who produced them, how they were produced, or whether a model can load them.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import struct
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

import numpy as np

from .nodes import Dependency, Node

TEXT_DOMAIN = b"nsllm:a1-text-rows:v1\0"
ARRAY_DOMAIN = b"nsllm:a1-token-blocks:v1\0"
PREPROCESSING = "join-tokenize-drop-tail-int32-reshape/v1"
MAX_ARCHIVE_BYTES = 1024**3
MAX_MEMBERS = 4096
MAX_CENTRAL_DIRECTORY_BYTES = 16 * 1024**2
MAX_MEMBER_BYTES = 4 * 1024**3
MAX_TOTAL_UNCOMPRESSED = 8 * 1024**3
MAX_COMPRESSION_RATIO = 1000
MAX_NPY_HEADER = 64 * 1024
MAX_METADATA_BYTES = 1024 * 1024
_CHUNK = 1024 * 1024


def _semantic(name: str, projection: str, value: dict[str, Any]) -> Node:
    return Node(
        kind="semantic_content",
        name=name,
        payload={"projection": projection, "version": 1, "value": value},
    )


def dataset_selector_node(name: str, selector: Mapping[str, Any]) -> Node:
    """Identify A1's declared selection and preprocessing context, not data."""
    required = {
        "dataset",
        "dataset_config",
        "text_field",
        "train_split",
        "eval_split",
        "test_split",
        "separator",
        "tokenizer",
        "seq_len",
        "preprocessing",
    }
    if not name or not isinstance(selector, Mapping) or set(selector) != required:
        raise ValueError(f"A1 dataset selector must contain exactly {sorted(required)}")
    for key in (
        "dataset",
        "text_field",
        "train_split",
        "eval_split",
        "test_split",
        "tokenizer",
        "preprocessing",
    ):
        if not isinstance(selector[key], str) or not selector[key].strip():
            raise ValueError(f"Invalid A1 dataset selector field: {key}")
    if selector["dataset_config"] is not None and (
        not isinstance(selector["dataset_config"], str)
        or not selector["dataset_config"].strip()
    ):
        raise ValueError("dataset_config must be a nonempty string or null")
    if not isinstance(selector["separator"], str):
        raise ValueError("separator must be a string")
    if type(selector["seq_len"]) is not int or selector["seq_len"] < 1:
        raise ValueError("seq_len must be a positive integer")
    if selector["preprocessing"] != PREPROCESSING:
        raise ValueError(
            f"Unsupported A1 preprocessing recipe: {selector['preprocessing']!r}"
        )
    return _semantic(name, "a1-dataset-selector", dict(selector))


def ordered_text_rows_node(name: str, rows: Iterable[str]) -> Node:
    """Hash ordered UTF-8 rows with uint64 length framing and an explicit count."""
    digest = hashlib.sha256(TEXT_DOMAIN)
    count = 0
    for row in rows:
        if not isinstance(row, str):
            raise ValueError("Dataset text rows must be strings")
        try:
            encoded = row.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError("Dataset text row contains a lone surrogate") from error
        if len(encoded) >= 2**64:
            raise ValueError("Dataset text row is too large")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        count += 1
    return _semantic(
        name,
        "a1-ordered-text-rows",
        {"row_count": count, "framed_sha256": digest.hexdigest()},
    )


def token_blocks_node(name: str, blocks: Any) -> Node:
    """Identify a 2-D signed int32 block array in canonical little-endian C order."""
    array = np.asarray(blocks)
    if array.ndim != 2:
        raise ValueError("Token blocks must be a two-dimensional array")
    if array.dtype.kind != "i" or array.dtype.itemsize != 4:
        raise ValueError("Token blocks must have signed int32 dtype")
    # Canonicalize equivalent native/endian/layout representations to one array
    # identity; the canonical dtype, byte order, and layout are recorded below.
    canonical = np.ascontiguousarray(array.astype(np.dtype("<i4"), copy=False))
    raw = b"" if canonical.nbytes == 0 else memoryview(canonical).cast("B")
    digest = hashlib.sha256(ARRAY_DOMAIN)
    digest.update(struct.pack(">Q", array.ndim))
    for dimension in array.shape:
        digest.update(struct.pack(">Q", dimension))
    digest.update(b"<i4\0C\0")
    digest.update(raw)
    return _semantic(
        name,
        "a1-token-blocks",
        {
            "shape": list(array.shape),
            "dtype": "int32",
            "byte_order": "little",
            "layout": "C",
            "byte_length": canonical.nbytes,
            "canonical_sha256": digest.hexdigest(),
        },
    )


def dataset_observation_node(
    name: str, selector: Node, rows: Node, blocks: Node
) -> Node:
    """Compose selector, observed rows, and prepared blocks as distinct inputs."""
    if any(node.kind != "semantic_content" for node in (selector, rows, blocks)):
        raise ValueError("Dataset observations require semantic content nodes")
    selector_payload = cast(dict[str, Any], selector.to_dict()["payload"])
    selector_value = selector_payload["value"]
    row_payload = cast(dict[str, Any], rows.to_dict()["payload"])
    block_payload = cast(dict[str, Any], blocks.to_dict()["payload"])
    if (
        selector_payload["projection"] != "a1-dataset-selector"
        or row_payload["projection"] != "a1-ordered-text-rows"
        or block_payload["projection"] != "a1-token-blocks"
        or selector_payload["version"] != 1
        or row_payload["version"] != 1
        or block_payload["version"] != 1
        or not isinstance(selector_value, dict)
        or not isinstance(row_payload["value"], dict)
        or not isinstance(block_payload["value"], dict)
    ):
        raise ValueError("Dataset observation nodes have incorrect projections")
    shape = block_payload["value"].get("shape")
    seq_len = selector_value.get("seq_len")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(size) is not int or size < 0 for size in shape)
        or type(seq_len) is not int
        or seq_len < 1
    ):
        raise ValueError("Invalid dataset selector or token block shape")
    if shape[1] != seq_len:
        raise ValueError("Prepared token block width differs from selector seq_len")
    return Node(
        kind="composite",
        name=name,
        payload={
            "recipe": {"name": "neural/a1/dataset-observation", "version": 1},
            "purpose": "component",
        },
        dependencies=(
            Dependency("uses-semantic", "selector", selector.id),
            Dependency("uses-semantic", "text-rows", rows.id),
            Dependency("uses-semantic", "token-blocks", blocks.id),
        ),
    )


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Mirror stages.results._resolve_archive_path fallback behavior."""
    candidate_path = Path(path)
    if not candidate_path.is_file():
        candidate = (
            candidate_path
            if candidate_path.suffix == ".npz"
            else candidate_path.with_suffix(".npz")
        )
        if candidate.is_file():
            return candidate
        if candidate.parent.name == "params":
            companion = candidate.parent.parent / "checkpoints" / candidate.name
            if companion.is_file():
                return companion
        elif candidate.parent.name == "checkpoints":
            companion = candidate.parent.parent / "params" / candidate.name
            if companion.is_file():
                return companion
    return candidate_path


@dataclass(frozen=True)
class CheckpointInspection:
    path: str
    sha256: str
    byte_length: int
    classification: str
    parameters: tuple[dict[str, Any], ...]
    optimizer_members: tuple[str, ...]
    metadata_present: bool
    metadata_fields: tuple[str, ...] | None
    provenance: str = "unknown"
    runtime_compatibility: str = "not_established"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "classification": self.classification,
            "parameters": list(self.parameters),
            "optimizer_members": list(self.optimizer_members),
            "metadata_present": self.metadata_present,
            "metadata_fields": (
                None if self.metadata_fields is None else list(self.metadata_fields)
            ),
            "provenance": self.provenance,
            "runtime_compatibility": self.runtime_compatibility,
        }


@dataclass(frozen=True)
class CheckpointByteVerification:
    path: str
    sha256: str
    byte_length: int
    expected_digest: str
    expected_length: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "expected_digest": self.expected_digest,
            "expected_length": self.expected_length,
        }


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _hash_stream(stream: IO[bytes]) -> tuple[str, int]:
    digest = hashlib.sha256()
    length = 0
    stream.seek(0)
    while chunk := stream.read(_CHUNK):
        digest.update(chunk)
        length += len(chunk)
    return digest.hexdigest(), length


def hash_checkpoint_file(path: str | Path) -> tuple[str, int]:
    """Stream exact bytes of a regular file without interpreting its format."""
    selected = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(selected, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Checkpoint must be a regular file")
        first = _hash_stream(stream)
        second = _hash_stream(stream)
        after = os.fstat(stream.fileno())
        try:
            path_info = selected.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError("Checkpoint path changed while hashing") from error
        if (
            first != second
            or _signature(before) != _signature(after)
            or _signature(after) != _signature(path_info)
        ):
            raise ValueError("Checkpoint changed while hashing")
        return first


def _read_npy_header(member: IO[bytes]) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(member)
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError("Unsupported NPY version")
    prefix = member.read(2 if version == (1, 0) else 4)
    length = int.from_bytes(prefix, "little")
    if length > MAX_NPY_HEADER:
        raise ValueError("Declared NPY header exceeds inspection limit")
    # NumPy's size check follows its read. Bound the read before handing it off.
    member = io.BytesIO(prefix + member.read(length))
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(
            member, max_header_size=MAX_NPY_HEADER
        )
    elif version in ((2, 0), (3, 0)):
        # NumPy's v2 reader uses the same length framing; header ASCII/UTF-8
        # handling is selected by read_magic in current supported NumPy versions.
        if version == (3, 0):
            read_header = getattr(np.lib.format, "_read_array_header", None)
            if read_header is None:
                raise ValueError("NumPy cannot safely inspect NPY v3 headers")
            shape, fortran, dtype = read_header(
                member, version, max_header_size=MAX_NPY_HEADER
            )
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                member, max_header_size=MAX_NPY_HEADER
            )
    else:
        raise ValueError(f"Unsupported NPY format version: {version}")
    if dtype.hasobject:
        raise ValueError(
            "Object arrays are not inspected because they may contain pickle"
        )
    if any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise ValueError("Invalid array shape in checkpoint")
    return tuple(shape), bool(fortran), dtype


def _inspect_zip(
    stream: IO[bytes], archive_size: int
) -> tuple[
    str, tuple[dict[str, Any], ...], tuple[str, ...], bool, tuple[str, ...] | None
]:
    if archive_size > MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"Checkpoint exceeds {MAX_ARCHIVE_BYTES} byte inspection limit"
        )
    # zipfile materializes every ZipInfo when opened, so bound that allocation
    # before constructing ZipFile.
    stream.seek(max(0, archive_size - (65535 + 22)))
    tail = stream.read(65535 + 22)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0 or len(tail) - eocd < 22:
        raise ValueError("Missing or truncated ZIP end-of-central-directory record")
    (
        _,
        disk,
        central_disk,
        disk_entries,
        entry_count,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd)
    if len(tail) - eocd != 22 + comment_size:
        raise ValueError("Malformed ZIP comment or trailing archive bytes")
    if disk or central_disk or disk_entries != entry_count:
        raise ValueError("Multi-disk ZIP checkpoints are unsupported")
    if (
        entry_count == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        raise ValueError("ZIP64 checkpoint archives are outside the inspection limits")
    if entry_count > MAX_MEMBERS:
        raise ValueError(f"Checkpoint has more than {MAX_MEMBERS} members")
    if central_size > MAX_CENTRAL_DIRECTORY_BYTES:
        raise ValueError("Checkpoint central directory exceeds inspection limit")
    eocd_offset = archive_size - len(tail) + eocd
    if central_offset + central_size != eocd_offset:
        raise ValueError("Invalid ZIP central directory bounds")
    if eocd_offset >= 20:
        stream.seek(eocd_offset - 20)
        if stream.read(4) == b"PK\x06\x07":
            raise ValueError("ZIP64 locator outside inspection limits")
    stream.seek(central_offset)
    for _ in range(entry_count):
        header = stream.read(46)
        if len(header) != 46 or header[:4] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central directory entry")
        stream.seek(sum(struct.unpack_from("<3H", header, 28)), os.SEEK_CUR)
        if stream.tell() > eocd_offset:
            raise ValueError("ZIP entry outside central directory")
    if stream.tell() != eocd_offset:
        raise ValueError("ZIP member count differs from directory declaration")
    with zipfile.ZipFile(stream, "r") as archive:
        entries = archive.infolist()
        if len(entries) != entry_count:
            raise ValueError("ZIP member count differs from directory declaration")
        if len(entries) > MAX_MEMBERS:
            raise ValueError(f"Checkpoint has more than {MAX_MEMBERS} members")
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Checkpoint contains duplicate ZIP member names")
        total = 0
        for entry in entries:
            path = Path(entry.filename)
            if (
                entry.orig_filename != entry.filename
                or path.is_absolute()
                or "\\" in entry.filename
                or any(part in ("", ".", "..") for part in entry.filename.split("/"))
                or entry.is_dir()
            ):
                raise ValueError(
                    "Checkpoint contains an unsafe or non-file member name"
                )
            if entry.flag_bits & 0x1:
                raise ValueError("Encrypted ZIP members are unsupported")
            if entry.file_size > MAX_MEMBER_BYTES:
                raise ValueError("Checkpoint member exceeds decompression limit")
            total += entry.file_size
            if total > MAX_TOTAL_UNCOMPRESSED:
                raise ValueError("Checkpoint exceeds total decompression limit")
            if entry.file_size and entry.compress_size == 0:
                raise ValueError("Invalid ZIP compression size")
            if (
                entry.compress_size
                and entry.file_size / entry.compress_size > MAX_COMPRESSION_RATIO
            ):
                raise ValueError("Checkpoint member exceeds compression-ratio limit")

        keys: set[str] = set()
        parameter_info: list[dict[str, Any]] = []
        optimizer = []
        metadata_fields = None
        metadata_present = False
        for entry in entries:
            if not entry.filename.endswith(".npy"):
                raise ValueError(
                    f"Unexpected non-NPY checkpoint member: {entry.filename}"
                )
            key = entry.filename[:-4]
            if not key or key in keys:
                raise ValueError("Checkpoint contains ambiguous array keys")
            keys.add(key)
            with archive.open(entry, "r") as member:
                shape, fortran, dtype = _read_npy_header(member)
                expected_body = int(np.prod(shape, dtype=object)) * dtype.itemsize
                header_position = member.tell()
                if expected_body + header_position != entry.file_size:
                    raise ValueError(f"Malformed NPY member size: {entry.filename}")
                remaining = expected_body
                body_hash = hashlib.sha256()
                while remaining:
                    chunk = member.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise ValueError(f"Truncated NPY member: {entry.filename}")
                    remaining -= len(chunk)
                    body_hash.update(chunk)
                if member.read(1):
                    raise ValueError(f"Unexpected trailing NPY data: {entry.filename}")
            item = {
                "key": key,
                "shape": list(shape),
                "dtype": dtype.str,
                "fortran_order": fortran,
                "data_sha256": body_hash.hexdigest(),
                "data_byte_length": expected_body,
            }
            if key.startswith("params/") or re.fullmatch(r"[^/]+/(A|B)", key):
                parameter_info.append(item)
            if key.startswith("opt/") or key == "__opt_treedef__":
                optimizer.append(key)
            if key == "__metadata__":
                metadata_present = True
                if (
                    expected_body <= MAX_METADATA_BYTES
                    and dtype.kind in "US"
                    and shape == ()
                ):
                    raw: bytes
                    with archive.open(entry, "r") as metadata_member:
                        _read_npy_header(metadata_member)
                        raw = metadata_member.read(MAX_METADATA_BYTES + 1)
                    # The array payload has a bounded scalar string representation;
                    # parse only JSON text, never object/pickle values.
                    if dtype.kind == "U":
                        width = dtype.itemsize // 4
                        encoding = (
                            "utf-32-be" if dtype.byteorder == ">" else "utf-32-le"
                        )
                        try:
                            text = raw.decode(encoding)[:width].rstrip("\x00")
                        except UnicodeDecodeError:
                            text = ""
                    else:
                        text = raw.decode("utf-8", errors="strict").rstrip("\x00")
                    try:
                        parsed = json.loads(text)
                    except (ValueError, UnicodeDecodeError):
                        parsed = None
                    if isinstance(parsed, dict) and all(
                        isinstance(key, str) for key in parsed
                    ):
                        metadata_fields = tuple(sorted(parsed))

        full = any(key.startswith("params/") for key in keys)
        prefix = "params/" if full else ""
        params = {
            key for key in keys if re.fullmatch(prefix + r"layer_[0-9]+/[AB]", key)
        }
        paired = bool(params) and all(
            key[:-1] + ("A" if key[-1] == "B" else "B") in params for key in params
        )
        extras = keys - params
        weights = paired and not extras and not full
        full = (
            full
            and paired
            and all(
                key in {"__metadata__", "__opt_treedef__"}
                or re.fullmatch(r"opt/leaf_[0-9]+", key)
                for key in extras
            )
        )
        classification = (
            "full_a1" if full else "weights_only_a1" if weights else "unknown"
        )
        return (
            classification,
            tuple(sorted(parameter_info, key=lambda x: x["key"])),
            tuple(sorted(optimizer)),
            metadata_present,
            metadata_fields,
        )


def inspect_a1_checkpoint(path: str | Path) -> CheckpointInspection:
    """Hash and structurally inspect an NPZ without loading arrays or pickle."""
    selected = resolve_checkpoint_path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(selected, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Checkpoint must be a regular file")
        if before.st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Checkpoint exceeds raw-file inspection limit")
        first_digest, first_length = _hash_stream(stream)
        stream.seek(0)
        try:
            inspected = _inspect_zip(stream, first_length)
        except (
            zipfile.BadZipFile,
            EOFError,
            OSError,
            ValueError,
            RuntimeError,
            struct.error,
            OverflowError,
        ) as error:
            raise ValueError(
                f"Malformed or unsafe checkpoint archive: {error}"
            ) from error
        second_digest, second_length = _hash_stream(stream)
        after = os.fstat(stream.fileno())
        try:
            path_info = selected.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError("Checkpoint path changed during inspection") from error
        if (
            first_digest != second_digest
            or first_length != second_length
            or _signature(before) != _signature(after)
            or _signature(after) != _signature(path_info)
        ):
            raise ValueError("Checkpoint changed during inspection")
    classification, parameters, optimizer, has_metadata, metadata_fields = inspected
    return CheckpointInspection(
        path=selected.name,
        sha256=first_digest,
        byte_length=first_length,
        classification=classification,
        parameters=parameters,
        optimizer_members=optimizer,
        metadata_present=has_metadata,
        metadata_fields=metadata_fields,
    )


def verify_a1_checkpoint(
    path: str | Path, *, expected_digest: str, expected_length: int | None = None
) -> CheckpointByteVerification:
    """Verify exact bytes only; no archive parsing or producer claim is made."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("Expected digest must be lowercase SHA-256 hex")
    if expected_length is not None and (
        type(expected_length) is not int or expected_length < 0
    ):
        raise ValueError("Expected byte length must be a nonnegative integer")
    selected = resolve_checkpoint_path(path)
    digest, byte_length = hash_checkpoint_file(selected)
    if digest != expected_digest:
        raise ValueError("Checkpoint SHA-256 mismatch")
    if expected_length is not None and byte_length != expected_length:
        raise ValueError("Checkpoint byte length mismatch")
    return CheckpointByteVerification(
        selected.name,
        digest,
        byte_length,
        expected_digest,
        expected_length,
    )


def imported_checkpoint_node(name: str, observation: CheckpointInspection) -> Node:
    """Represent supplied checkpoint bytes as an imported source observation."""
    return Node(
        kind="source",
        name=name,
        payload={"sha256": observation.sha256, "byte_length": observation.byte_length},
    )


__all__ = [
    "CheckpointByteVerification",
    "CheckpointInspection",
    "dataset_observation_node",
    "dataset_selector_node",
    "imported_checkpoint_node",
    "inspect_a1_checkpoint",
    "hash_checkpoint_file",
    "ordered_text_rows_node",
    "resolve_checkpoint_path",
    "token_blocks_node",
    "verify_a1_checkpoint",
]
