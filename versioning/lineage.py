"""Recipe-checked joins over completed stage records and their captured bytes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .execution import _file_identity, validate_symbolic_run
from .mining import validate_release
from .nodes import Node, reachable_nodes
from .registry import Registry
from .symbolic import validate_symbolic_definition
from .workflow import composite, slots, validate_run

JOIN = "workflow/linked-run"
PARSER = "symbolic/m3/parser-export/run"
EXPORT = "symbolic/m3/adapter-export/run"
MINER = "workflow/miner/run"
EMBED = "workflow/embedding/run"
# consumer, input slot, producer, output slot
PORTS = {
    (EXPORT, "input", PARSER, "accepted"),
    (MINER, "documents.jsonl", EXPORT, "accepted"),
    (MINER, "bundle.json", EXPORT, "ontology-bundle"),
    (EMBED, "documents.jsonl", EXPORT, "accepted"),
    (EMBED, "documents.jsonl", MINER, "accepted.jsonl"),
    (EMBED, "patterns.jsonl", MINER, "patterns.jsonl"),
    ("workflow/student-train/run", "train.jsonl", PARSER, "student-pairs"),
    ("workflow/student-train/run", "validation.jsonl", PARSER, "student-pairs"),
}


def underlying(node: Node, index: Mapping[str, Node]) -> Node:
    while node.name == JOIN:
        node = slots(node, index)["consumer"]
    if node.name == "workflow/parser/observed":
        node = slots(node, index)["run"]
    return node


def validate_link(node: Node, index: Mapping[str, Node]) -> None:
    edge = slots(node, index)
    if "consumer" not in edge or len(edge) < 2 or node != composite(JOIN, "run", edge):
        raise ValueError("Invalid workflow link")
    consumer = underlying(edge.pop("consumer"), index)
    inv = slots(consumer, index).get("inputs")
    if inv is None:
        raise ValueError("Consumer lacks invocation")
    inputs = slots(inv, index)
    for port, record in edge.items():
        producer = underlying(record, index)
        allowed = [
            output
            for c, i, p, output in PORTS
            if (c, i, p) == (consumer.name, port, producer.name)
        ]
        if (
            consumer.name == "workflow/a1-eval/run"
            and port == "checkpoint.npz"
            and producer.name == "workflow/a1-train/run"
        ):
            allowed = [
                k
                for k in slots(producer, index)
                if k.startswith(("checkpoints/", "params/"))
            ]
        if (
            consumer.name == "workflow/student-eval/run"
            and port.startswith("model/")
            and producer.name == "workflow/student-train/run"
        ):
            allowed = [port.removeprefix("model/")]
        source = inputs.get(port)
        imports = {
            (EXPORT, "input"): "parser-results",
            (MINER, "documents.jsonl"): "documents",
            (EMBED, "documents.jsonl"): "documents",
            (EMBED, "patterns.jsonl"): "patterns",
            ("workflow/a1-eval/run", "checkpoint.npz"): "checkpoint",
        }
        if (
            producer.kind == "source"
            and (consumer.name, port) in imports
            and producer.name == "import/" + imports[consumer.name, port]
            and source is not None
            and source.payload == producer.payload
        ):
            continue
        outputs = slots(producer, index)
        if (
            source is None
            or source.kind != "source"
            or not any(
                k in outputs
                and outputs[k].kind == "artifact"
                and outputs[k].payload == source.payload
                for k in allowed
            )
        ):
            raise ValueError(f"Incompatible or changed artifact at {port}")


def validate_known(nodes: tuple[Node, ...]) -> None:
    index = {n.id: n for n in nodes}
    for node in nodes:
        # Validate each run within its own closure, including both sides of joins.
        if node.name in (PARSER, EXPORT):
            validate_symbolic_run(nodes, node.id)
        elif node.name.startswith("workflow/") and node.name.endswith("/run"):
            validate_run(nodes, node.id)
        elif node.name == JOIN:
            validate_link(node, index)
        elif node.name in ("workflow/parser/failed", "workflow/parser/observed"):
            from .parser_observation import validate_parser

            validate_parser(node, index)
        elif node.name == "workflow/miner/installation":
            from .mining import validate_installation

            validate_installation(node, index)
        elif node.name == "workflow/miner/release":
            validate_release(reachable_nodes((node.id,), index.__getitem__))
    root = nodes[-1]
    if root.name == "symbolic/definition":
        validate_symbolic_definition(nodes, root.id)
    elif root.name == "neural/a1/definition":
        from .neural import validate_a1_definition

        validate_a1_definition(nodes, root.id)


def link(registry: Registry, consumer: str, upstream: dict[str, str]) -> Node:
    if "consumer" in upstream:
        raise ValueError("Reserved input name")
    root = composite(
        JOIN,
        "run",
        {
            "consumer": registry.get(consumer),
            **{k: registry.get(v) for k, v in upstream.items()},
        },
    )
    closure = reachable_nodes(
        (root.id,), lambda key: root if key == root.id else registry.get(key)
    )
    validate_known(closure)
    registry.register_batch([root])
    return root


def verify_file(nodes: tuple[Node, ...], node_id: str, path: Path) -> None:
    node = next((n for n in nodes if n.id == node_id), None)
    if (
        node is None
        or node.kind not in ("source", "artifact")
        or _file_identity(path) != node.payload
    ):
        raise ValueError("File differs from manifest artifact")
