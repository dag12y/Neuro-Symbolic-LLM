"""Local append-only storage for component records."""

import fcntl
import os
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile

from .hashing import canonical_json, load_json
from .records import (
    ComponentRecord,
    check_release_labels,
    dependency_order,
    validate_digest,
)


class Registry:
    def __init__(self, root: str | Path = ".version-registry") -> None:
        self.root = Path(root).absolute()
        self._records = self.root / "records"

    def get(self, record_id: str) -> ComponentRecord:
        """Read and verify an exact record. A missing ID raises FileNotFoundError."""
        validate_digest(record_id)
        path = self._records / f"{record_id}.json"
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"Record is not a regular file: {record_id}")
            with open(fd, "rb", closefd=False) as stream:
                record = ComponentRecord.from_dict(load_json(stream.read()))
        finally:
            os.close(fd)
        if record.record_id != record_id:
            raise ValueError(
                f"Record identity does not match its filename: {record_id}"
            )
        return record

    def _snapshot(self) -> dict[str, ComponentRecord]:
        try:
            paths = sorted(
                path for path in self._records.iterdir() if path.name.endswith(".json")
            )
        except FileNotFoundError:
            return {}
        return {path.stem: self.get(path.stem) for path in paths}

    def list_records(
        self,
        *,
        kind: str | None = None,
        name: str | None = None,
        label: str | None = None,
    ) -> list[ComponentRecord]:
        """Read verified records in ID order with optional exact filters."""
        return [
            record
            for record in self._snapshot().values()
            if (kind is None or record.kind == kind)
            and (name is None or record.name == name)
            and (label is None or record.label == label)
        ]

    def find(
        self, kind: str, name: str, label: str | None = None
    ) -> list[ComponentRecord]:
        """Find matching records in record ID order."""
        return self.list_records(kind=kind, name=name, label=label)

    def verify_integrity(self) -> list[ComponentRecord]:
        """Verify record identities, release labels and all dependency graphs."""
        records = self._snapshot()
        check_release_labels(records.values())
        for record_id in records:
            dependency_order(record_id, records)
        return list(records.values())

    def register(self, record: ComponentRecord) -> str:
        """Validate provenance and atomically publish an immutable record."""
        record = ComponentRecord.from_dict(record.to_dict())
        self._records.mkdir(parents=True, exist_ok=True)
        with (self.root / ".write.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = self._snapshot()
            existing = records.get(record.record_id)
            records[record.record_id] = record
            dependency_order(record.record_id, records)
            check_release_labels(records.values())
            if existing is None:
                self._publish(record)
        return record.record_id

    def _publish(self, record: ComponentRecord) -> None:
        destination = self._records / f"{record.record_id}.json"
        with NamedTemporaryFile(dir=self._records, suffix=".tmp") as staged:
            staged.write(canonical_json(record.to_dict()) + b"\n")
            staged.flush()
            os.fsync(staged.fileno())
            os.link(staged.name, destination)
        directory_fd = os.open(self._records, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
