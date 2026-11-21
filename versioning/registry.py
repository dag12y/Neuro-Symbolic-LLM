"""Transactional, immutable node storage in a local SQLite database."""

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

from .hashing import canonical_json, load_json
from .nodes import Node, reachable_nodes, validate_id

_CREATE_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT NOT NULL PRIMARY KEY,
    canonical BLOB NOT NULL
)
"""


def _stored_bytes(connection: sqlite3.Connection, node_id: str) -> bytes | None:
    row = connection.execute(
        "SELECT canonical FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], bytes):
        raise ValueError(f"Registry node is not stored as bytes: {node_id}")
    return row[0]


def _get(connection: sqlite3.Connection, node_id: str) -> Node:
    data = _stored_bytes(connection, node_id)
    if data is None:
        raise KeyError(node_id)
    node = Node.from_dict(load_json(data))
    if node.id != node_id or canonical_json(node.to_dict()) != data:
        raise ValueError(f"Registry node does not match its ID or bytes: {node_id}")
    return node


class Registry:
    """Store immutable nodes, opening a separate connection for each operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).absolute()

    def get(self, node_id: str) -> Node:
        """Read and verify a node by full ID. An absent node raises KeyError."""
        validate_id(node_id)
        if not self.path.is_file():
            raise FileNotFoundError(f"Registry database not found: {self.path}")
        uri = self.path.as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            return _get(connection, node_id)

    def register_batch(self, nodes: Iterable[Node]) -> tuple[str, ...]:
        """Publish a batch atomically and return its unique IDs in input order."""
        pending: dict[str, Node] = {}
        canonical: dict[str, bytes] = {}
        for node in nodes:
            data = node.to_dict()
            verified = Node.from_dict(data)
            encoded = canonical_json(data)
            if canonical.setdefault(verified.id, encoded) != encoded:
                raise ValueError(f"Conflicting data for node ID: {verified.id}")
            pending[verified.id] = verified
        if not pending:
            return ()

        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.execute(_CREATE_NODES)
            with connection:
                connection.execute("BEGIN IMMEDIATE")

                def lookup(node_id: str) -> Node:
                    if node_id in pending:
                        return pending[node_id]
                    return _get(connection, node_id)

                for node in reachable_nodes(pending, lookup):
                    if node.id not in pending:
                        continue
                    encoded = canonical[node.id]
                    stored = _stored_bytes(connection, node.id)
                    if stored is None:
                        connection.execute(
                            "INSERT INTO nodes (id, canonical) VALUES (?, ?)",
                            (node.id, encoded),
                        )
                    elif stored != encoded:
                        raise ValueError(f"Conflicting data for node ID: {node.id}")
        return tuple(pending)
