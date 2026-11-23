"""Observe stage execution in a fresh workspace and verify typed workflow joins."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from .execution import _file_identity
from .nodes import Dependency, Node, reachable_nodes
from .registry import Registry

PROJECT = Path(__file__).resolve().parents[1]
MINER_FILES = (
    "accepted.jsonl",
    "rejected.jsonl",
    "facts.metta",
    "metadata.metta",
    "provenance.jsonl",
    "admitted.metta",
    "admission.json",
    "quarantine.jsonl",
    "support.jsonl",
    "exact.jsonl",
    "exact-rejected.jsonl",
    "patterns.jsonl",
    "patterns-rejected.jsonl",
)
INPUTS = {
    "miner": {"documents.jsonl", "bundle.json"},
    "embedding": {"documents.jsonl", "patterns.jsonl"},
    "a1-train": {"config.yaml"},
    "a1-eval": {"config.yaml", "checkpoint.npz"},
    "student-train": {"config.yaml", "train.jsonl"},
    "student-eval": {"sentences.txt"},
}


def composite(name: str, purpose: str, slots: Mapping[str, Node]) -> Node:
    roles = {
        "source": "uses-source",
        "artifact": "uses-artifact",
        "semantic_content": "uses-semantic",
        "composite": "uses-component",
        "external_release": "uses-release",
    }
    return Node(
        kind="composite",
        name=name,
        payload={"recipe": {"name": name, "version": 1}, "purpose": purpose},
        dependencies=tuple(
            Dependency(roles[n.kind], slot, n.id) for slot, n in slots.items()
        ),
    )


def observation(name: str, value: dict[str, Any]) -> Node:
    return Node(
        kind="semantic_content",
        name=name,
        payload={"projection": name, "version": 1, "value": value},
    )


def slots(node: Node, index: Mapping[str, Node]) -> dict[str, Node]:
    result = {e.slot: index[e.target_id] for e in node.dependencies}
    if len(result) != len(node.dependencies):
        raise ValueError("Conflicting edge slots")
    return result


def inputs_for(stage: str, names: set[str]) -> None:
    if any(
        not n or any(part in ("", ".", "..") for part in n.split("/")) or "\\" in n
        for n in names
    ):
        raise ValueError("Unsafe logical input filename")
    required = INPUTS[stage]
    extra = names - required
    if required - names or (stage == "student-eval" and not extra):
        raise ValueError("Missing stage input")
    if (
        stage == "student-eval"
        and any(n.endswith(".safetensors") for n in extra)
        and all(n.startswith("model/") for n in extra)
    ):
        return
    if stage == "student-train" and extra <= {"validation.jsonl"}:
        return
    if extra:
        raise ValueError("Unexpected stage input")


def outputs_for(stage: str, names: set[str], definition: tuple[Node, ...]) -> None:
    required = {
        "embedding": {"embeddings.jsonl"},
        "miner": set(MINER_FILES),
        "a1-eval": {"evaluation.json"},
        "student-eval": {"evaluation.txt"},
    }.get(stage, set())
    if stage == "a1-train":
        data = next(n for n in definition if n.name == "neural/a1/config/data")
        domains = cast(dict[str, Any], data.to_dict()["payload"])["value"]["domains"]
        required = {"config.yaml", "meta.json", "metrics.jsonl", "results.json"}
        for i, domain in enumerate(domains):
            for folder in ("params", "checkpoints"):
                required.add(f"{folder}/task_{i}_{domain['name']}.npz")
    if stage == "student-train":
        if (
            {"config.json", "training_metadata.json"} <= names
            and all("/" not in n for n in names)
            and any(n.endswith(".safetensors") for n in names)
        ):
            return
        raise ValueError("Incomplete student checkpoint")
    if names != required:
        raise ValueError(f"Unexpected {stage} output set: {sorted(names ^ required)}")


def producer_sources(stage: str) -> tuple[str, ...]:
    modules = ["workflow", "workflow_cli", "execution"]
    modules.append("mining" if stage == "miner" else "workflow_stages")
    if stage.startswith("a1-"):
        modules.append("neural_integrity")
    return tuple(f"versioning/{n}.py" for n in modules)


def capture(
    stage: str,
    registry: Registry,
    files: dict[str, Path],
    destination: Path,
    definition: Callable[[], tuple[Node, ...]],
    settings: dict[str, Any],
    run: Callable[[Path, Path, dict[str, Any]], None],
) -> tuple[Node, bool]:
    inputs_for(stage, set(files))
    prefix = f"workflow/{stage}"
    if (
        destination.exists()
        or destination.is_symlink()
        or registry.path.resolve().is_relative_to(destination.resolve())
    ):
        raise ValueError("Use a fresh output directory separate from the registry")
    database = registry.path.resolve()
    sqlite_files = [
        database,
        *(Path(str(database) + s) for s in ("-wal", "-shm", "-journal")),
    ]
    if any(
        p.exists() and f.exists() and p.samefile(f)
        for p in sqlite_files
        for f in files.values()
    ):
        raise ValueError("Registry and input files overlap")
    dest_parent = destination.parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    defs = definition()
    producer = {
        f"producer/{p}": Node(
            kind="source", name=p, payload=_file_identity(PROJECT / p)
        )
        for p in producer_sources(stage)
    }
    with TemporaryDirectory(dir=dest_parent) as tmp:
        scratch = Path(tmp)
        private = scratch / "inputs"
        out = scratch / "outputs"
        private.mkdir()
        out.mkdir()
        sources = {}
        for name, path in files.items():
            target = private / name
            target.parent.mkdir(parents=True, exist_ok=True)
            sources[name] = Node(
                kind="source",
                name=f"{prefix}/input/{name}",
                payload=_file_identity(path, target),
            )
        runtime: dict[str, Any] = {}
        failure = None
        partial = {}
        try:
            run(private, out, runtime)
            paths = {
                p.relative_to(out).as_posix(): p for p in out.rglob("*") if p.is_file()
            }
            if any(p.is_symlink() for p in out.rglob("*")):
                raise ValueError("Symlink in producer outputs")
            outputs_for(stage, set(paths), defs)
        except Exception as error:
            failure = type(error).__name__
            paths = {}
            if out.is_dir() and not out.is_symlink():
                partial = {
                    p.relative_to(out).as_posix(): _file_identity(p, follow_links=False)
                    for p in out.rglob("*")
                    if p.is_file() and not p.is_symlink()
                }

            print(f"{stage} failed: {error}", file=sys.stderr)
        if (
            any(
                _file_identity(PROJECT / n.name) != n.payload for n in producer.values()
            )
            or definition() != defs
            or any(
                _file_identity(path) != sources[name].payload
                or _file_identity(private / name) != sources[name].payload
                for name, path in files.items()
            )
        ):
            raise ValueError("Stage inputs changed during execution")
        options = observation(f"{prefix}/options", settings)
        observations = observation(f"{prefix}/runtime", runtime)
        invocation = composite(
            f"{prefix}/inputs",
            "component",
            {
                "definition": defs[-1],
                "options": options,
                "runtime": observations,
                **producer,
                **sources,
            },
        )
        outcome = observation(
            f"{prefix}/outcome",
            {
                "status": "failed" if failure else "completed",
                "error": failure,
                "partial_files": partial,
            },
        )
        artifacts = {
            name: Node(
                kind="artifact",
                name=f"{prefix}/output/{name}",
                payload=_file_identity(path),
                dependencies=(Dependency("produced-by", "inputs", invocation.id),),
            )
            for name, path in paths.items()
        }
        root = composite(
            f"{prefix}/run",
            "run",
            {"inputs": invocation, "outcome": outcome, **artifacts},
        )
        batch = (
            *defs,
            *producer.values(),
            *sources.values(),
            options,
            observations,
            invocation,
            outcome,
            *artifacts.values(),
            root,
        )
        validate_run(batch, root.id)
        if failure is None:
            _publish(destination, paths, artifacts)
        registry.register_batch(batch)
        return root, failure is None


def _publish(
    destination: Path, paths: dict[str, Path], artifacts: dict[str, Node]
) -> None:
    destination.mkdir(mode=0o700)
    flags = os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW
    with ExitStack() as stack:
        root = os.open(destination, flags)
        stack.callback(os.close, root)
        directories = {Path("."): root}
        for name, path in paths.items():
            parent = Path(".")
            for part in Path(name).parts[:-1]:
                child = parent / part
                if child not in directories:
                    os.mkdir(part, mode=0o700, dir_fd=directories[parent])
                    fd = os.open(part, flags, dir_fd=directories[parent])
                    stack.callback(os.close, fd)
                    directories[child] = fd
                parent = child
            os.link(path, Path(name).name, dst_dir_fd=directories[parent])
        for name, node in artifacts.items():
            if _file_identity(destination / name, follow_links=False) != node.payload:
                raise ValueError("Published artifact changed")
        if any(
            not os.path.samestat(os.fstat(fd), (destination / path).lstat())
            for path, fd in directories.items()
        ):
            raise ValueError("Published directory changed")


def validate_run(nodes: Iterable[Node], root_id: str) -> None:
    from .workflow_stages import validate_definition, validate_options

    index = {n.id: n for n in nodes}
    root = index[root_id]
    parts = root.name.split("/")
    if (
        len(parts) != 3
        or parts[0] != "workflow"
        or parts[1] not in INPUTS
        or parts[2] != "run"
    ):
        raise ValueError("Invalid workflow run name")
    stage = parts[1]
    prefix = f"workflow/{stage}"
    outputs = slots(root, index)
    try:
        inv = outputs.pop("inputs")
        outcome = outputs.pop("outcome")
        inputs = slots(inv, index)
        definition = inputs.pop("definition")
        options = inputs.pop("options")
        observations = inputs.pop("runtime")
        producer = {
            f"producer/{p}": inputs.pop(f"producer/{p}")
            for p in producer_sources(stage)
        }
    except KeyError as e:
        raise ValueError("Missing required stage slot") from e

    def value(n: Node, label: str) -> dict[str, Any]:
        data: Any = (
            cast(dict[str, Any], n.to_dict()["payload"]).get("value")
            if n.kind == "semantic_content"
            else None
        )
        if not isinstance(data, dict) or n != observation(f"{prefix}/{label}", data):
            raise ValueError("Invalid workflow projection")
        return data

    validate_options(stage, value(options, "options"))
    result = value(outcome, "outcome")
    runtime = value(observations, "runtime")
    if set(result) != {"status", "error", "partial_files"} or not isinstance(
        result["partial_files"], dict
    ):
        raise ValueError("Invalid workflow outcome")
    for name, identity in result["partial_files"].items():
        Node(kind="source", name=f"partial/{name}", payload=identity)
    if result["status"] == "completed" and result["partial_files"]:
        raise ValueError("Completed run includes partial files")
    if any(
        n.kind != "source" or k != "producer/" + n.name for k, n in producer.items()
    ):
        raise ValueError("Invalid producer implementation")
    inputs_for(stage, set(inputs))
    defs = reachable_nodes((definition.id,), index.__getitem__)
    validate_definition(stage, defs)
    if any(
        n.kind != "source" or n.name != f"{prefix}/input/{k}" for k, n in inputs.items()
    ):
        raise ValueError("Invalid stage input role")
    if inv != composite(
        f"{prefix}/inputs",
        "component",
        {
            "definition": definition,
            "options": options,
            "runtime": observations,
            **producer,
            **inputs,
        },
    ):
        raise ValueError("Invalid invocation")
    if result["status"] == "completed" and result["error"] is None:
        outputs_for(stage, set(outputs), defs)
        from .workflow_stages import validate_runtime

        validate_runtime(stage, runtime, defs, value(options, "options"))
    elif (
        result["status"] != "failed"
        or not isinstance(result["error"], str)
        or not result["error"]
        or outputs
    ):
        raise ValueError("Inconsistent workflow outcome")
    for name, n in outputs.items():
        if (
            n.kind != "artifact"
            or n.name != f"{prefix}/output/{name}"
            or n.dependencies != (Dependency("produced-by", "inputs", inv.id),)
        ):
            raise ValueError("Invalid artifact producer")
    if root != composite(
        f"{prefix}/run", "run", {"inputs": inv, "outcome": outcome, **outputs}
    ):
        raise ValueError("Invalid workflow run")
