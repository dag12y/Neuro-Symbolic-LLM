"""Extended reference-parser observations around the original M3 exporter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn, cast

from .execution import _PARSER_SUPPORT
from .nodes import Node, reachable_nodes, validate_id
from .registry import Registry
from .symbolic import validate_symbolic_definition
from .workflow import composite, observation, slots


class ParserExecutionError(ValueError):
    def __init__(self, root: Node):
        self.root = root
        super().__init__("Parser execution failed")


def observed_parser(run: Node, runtime: dict[str, object]) -> tuple[Node, ...]:
    obs = observation("workflow/parser/runtime", runtime)
    root = composite("workflow/parser/observed", "run", {"run": run, "runtime": obs})
    return obs, root


def failed_parser(
    registry: Registry,
    definition: tuple[Node, ...],
    support: tuple[Node, ...],
    source_name: str,
    identity: dict[str, object],
    options: dict[str, object],
    runtime: dict[str, object],
    error: Exception,
) -> NoReturn:
    src = Node(
        kind="source", name=f"workflow/parser/input/{source_name}", payload=identity
    )
    opts = observation("workflow/parser/options", options)
    obs = observation("workflow/parser/runtime", runtime)
    failure = observation("workflow/parser/error", {"exception": type(error).__name__})
    root = composite(
        "workflow/parser/failed",
        "run",
        {
            "definition": definition[-1],
            "input": src,
            "options": opts,
            "runtime": obs,
            "error": failure,
            **{n.name: n for n in support},
        },
    )
    batch = (*definition, *support, src, opts, obs, failure, root)
    validate_parser(root, {n.id: n for n in batch})
    registry.register_batch(batch)
    raise ParserExecutionError(root) from error


def validate_parser(root: Node, index: Mapping[str, Node]) -> None:
    edges = slots(root, index)
    if root != composite(root.name, "run", edges):
        raise ValueError("Invalid parser observation")

    def data(key: str) -> dict[str, Any]:
        n = edges[key]
        value = cast(dict[str, Any], n.to_dict()["payload"]).get("value")
        if not isinstance(value, dict) or n != observation(
            f"workflow/parser/{key}", value
        ):
            raise ValueError("Invalid parser observation projection")
        return value

    observed = root.name == "workflow/parser/observed"
    required = (
        {"run", "runtime"}
        if observed
        else {"definition", "input", "options", "runtime", "error", *_PARSER_SUPPORT}
    )
    if set(edges) != required:
        raise ValueError("Invalid parser slots")
    runtime = data("runtime")
    if (runtime or observed) and (
        set(runtime) != {"model", "backend", "revision_reported"}
        or not all(isinstance(runtime[k], str) for k in ("model", "backend"))
        or not isinstance(runtime["revision_reported"], str | None)
    ):
        raise ValueError("Invalid parser runtime")
    if observed:
        if edges["run"].name != "symbolic/m3/parser-export/run":
            raise ValueError("Expected reference parser export")
        return
    base = edges["definition"]
    validate_symbolic_definition(
        reachable_nodes((base.id,), index.__getitem__), base.id
    )
    if (
        edges["input"].kind != "source"
        or edges["input"].name
        not in (
            "workflow/parser/input/source.txt",
            "workflow/parser/input/source.jsonl",
        )
        or any(edges[k].kind != "source" or edges[k].name != k for k in _PARSER_SUPPORT)
    ):
        raise ValueError("Invalid parser implementation or source")
    options = data("options")
    flags = {
        "include_metta",
        "asserted_only",
        "student_pairs",
        "require_structured_output",
    }
    if set(options) != flags | {"corpus_id_sha256"} or any(
        type(options[k]) is not bool for k in flags
    ):
        raise ValueError("Invalid parser options")
    validate_id(options["corpus_id_sha256"])
    failure = data("error")
    if set(failure) != {"exception"} or not isinstance(failure["exception"], str):
        raise ValueError("Invalid parser failure")
