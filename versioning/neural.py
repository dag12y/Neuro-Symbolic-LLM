"""Offline Stage A1 definition recipe v1."""

from __future__ import annotations

import argparse
import math
import re
import struct
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from stages.config import (
    ACTIVATION_NAMES,
    A1Config,
    RunConfig,
    apply_overrides,
    parse_configs,
)

from .nodes import Dependency, Node, reachable_nodes
from .yaml_inputs import read_yaml

ROOT = "neural/a1/definition"
DEFAULT_CONFIG = Path("configs/stage_A/a1_lora_baseline.yaml")
PROJECT = Path(__file__).resolve().parents[1]
# Sources are whole modules. A shared file always designates the same leaf.
_SOURCES = {
    "substrate": (
        "frozenllm/__init__.py",
        "frozenllm/substrate/__init__.py",
        *(
            f"frozenllm/substrate/{name}.py"
            for name in (
                "architecture",
                "loader",
                "substrate",
                "interception",
                "drift",
                "memory",
                "torchax_backend",
                "torchax_models",
            )
        ),
        "metrics/performance.py",
        "metrics/__init__.py",
    ),
    "adapter": ("residual/__init__.py", "residual/adapter.py", "stages/config.py"),
    "data": ("stages/stage_base.py", "stages/config.py"),
    "optimization": (
        "stages/stage_A/a1.py",
        "stages/stage_A/__init__.py",
        "stages/stage_base.py",
        "stages/config.py",
    ),
    "evaluation": (
        "stages/stage_base.py",
        "metrics/performance.py",
        "metrics/representation.py",
    ),
    "driver": (
        "experiments/run_stage_a1.py",
        "stages/__init__.py",
        "stages/config.py",
        "stages/results.py",
    ),
}
_COMPONENTS = {
    "substrate": (),
    "adapter": ("substrate",),
    "data": ("substrate",),
    "optimization": ("adapter", "data"),
    "evaluation": ("adapter", "data"),
    "driver": ("optimization", "evaluation"),
}


def _number(value: object) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, float | int)
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite number")
    return struct.pack(">d", value).hex()


def _project(cfg: RunConfig) -> dict[str, dict[str, Any]]:
    raw = cfg.to_dict()
    adapter = raw["adapter"]
    for key in ("alpha", "init_std"):
        if adapter[key] is not None:
            adapter[key] = _number(adapter[key])
    if adapter["layers"] is not None:
        adapter["layers"] = sorted(adapter["layers"])
    t = cfg.training
    domains = raw["domains"]
    for d in domains:
        if d["epochs"] is None:
            d["epochs"] = t.epochs
    epochs = [d.pop("epochs") for d in domains]
    return {
        "substrate": {"model": t.model, "tokenizer": t.model},
        "adapter": adapter,
        "data": {"domains": domains, "seq_len": t.seq_len, "batch_size": t.batch_size},
        "optimization": {
            "optimizer": "adam",
            "learning_rate": _number(t.learning_rate),
            **{k: _number(v) for k, v in raw["objective"].items()},
            "epochs": epochs,
            "seed": t.seed,
            "reset_optimizer": t.reset_optimizer,
        },
        "evaluation": {"eval_batches": t.eval_batches},
        "driver": {"jit": t.jit, "log_every": t.log_every},
    }


def _keys(value: Mapping[str, Any], names: str) -> None:
    if set(value) != set(names.split()):
        raise ValueError(f"Expected A1 projection keys: {names}")


def _integer(value: Any, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"Expected integer >= {minimum}")


def _locator(value: Any) -> None:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", value)
        or any(p in (".", "..") for p in value.split("/"))
    ):
        raise ValueError("Expected a portable relative asset locator")


def _float_field(value: Any, *, positive: bool) -> None:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{16}", value):
        raise ValueError("Expected binary64 hex")
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError("Invalid A1 coefficient")


def _validate_values(values: dict[str, dict[str, Any]]) -> None:
    substrate, adapter, data = values["substrate"], values["adapter"], values["data"]
    optimizer, evaluation, driver = (
        values["optimization"],
        values["evaluation"],
        values["driver"],
    )
    _keys(substrate, "model tokenizer")
    _locator(substrate["model"])
    if substrate["model"] != substrate["tokenizer"]:
        raise ValueError("Tokenizer must use model locator")
    _keys(adapter, "rank activation alpha init_std late_start late_end layers seed")
    _integer(adapter["rank"], 1)
    _integer(adapter["seed"])
    if adapter["activation"] not in ACTIVATION_NAMES:
        raise ValueError("Unknown activation")
    for key in ("alpha", "init_std"):
        if adapter[key] is not None:
            _float_field(adapter[key], positive=True)
    for key in ("late_start", "late_end"):
        if adapter[key] is not None:
            _integer(adapter[key])
    if (
        adapter["late_start"] is not None
        and adapter["late_end"] is not None
        and adapter["late_end"] < adapter["late_start"]
    ):
        raise ValueError("Reversed layer range")
    layers = adapter["layers"]
    if layers is not None:
        if (
            not isinstance(layers, list)
            or not layers
            or any(adapter[k] is not None for k in ("late_start", "late_end"))
        ):
            raise ValueError("Invalid layer selection")
        for layer in layers:
            _integer(layer)
        if layers != sorted(set(layers)):
            raise ValueError("Expected sorted unique layers")
    _keys(data, "domains seq_len batch_size")
    _integer(data["seq_len"], 1)
    _integer(data["batch_size"], 1)
    domains = data["domains"]
    if not isinstance(domains, list) or not domains:
        raise ValueError("Missing ordered domains")
    names = set()
    for domain in domains:
        if not isinstance(domain, dict):
            raise ValueError("Invalid domain")
        _keys(
            domain,
            "name dataset dataset_config text_field train_split eval_split test_split separator",
        )
        name = domain["name"]
        if (
            not isinstance(name, str)
            or not re.fullmatch("[A-Za-z0-9_.-]+", name)
            or name in names
        ):
            raise ValueError("Invalid or duplicate domain name")
        names.add(name)
        _locator(domain["dataset"])
        for key in (
            "text_field",
            "train_split",
            "eval_split",
            "test_split",
            "dataset_config",
        ):
            if key == "dataset_config" and domain[key] is None:
                continue
            if not isinstance(domain[key], str) or not domain[key].strip():
                raise ValueError(f"Invalid domain {key}")
        if not isinstance(domain["separator"], str):
            raise ValueError("Invalid separator")
    _keys(
        optimizer,
        "optimizer learning_rate lambda_kl lambda_wd epochs seed reset_optimizer",
    )
    if optimizer["optimizer"] != "adam" or not isinstance(
        optimizer["reset_optimizer"], bool
    ):
        raise ValueError("Invalid optimizer policy")
    _integer(optimizer["seed"])
    for key in ("learning_rate", "lambda_kl", "lambda_wd"):
        _float_field(optimizer[key], positive=key == "learning_rate")
    if not isinstance(optimizer["epochs"], list) or len(optimizer["epochs"]) != len(
        domains
    ):
        raise ValueError("Expected epochs per ordered domain")
    for epochs in optimizer["epochs"]:
        _integer(epochs)
    _keys(evaluation, "eval_batches")
    _integer(evaluation["eval_batches"], 1)
    _keys(driver, "jit log_every")
    if not isinstance(driver["jit"], bool):
        raise ValueError("Invalid JIT policy")
    _integer(driver["log_every"], 1)


def _recipe(name: str) -> dict[str, Any]:
    return {
        "recipe": {"name": name, "version": 1},
        "purpose": "definition" if name == ROOT else "component",
    }


def _targets(group: str) -> dict[tuple[str, str], str]:
    if group == "definition":
        return {("uses-component", "driver"): "neural/a1/driver"}
    return {
        **{("uses-source", p): p for p in _SOURCES[group]},
        **{("uses-component", k): f"neural/a1/{k}" for k in _COMPONENTS[group]},
        ("uses-semantic", "configuration"): f"neural/a1/config/{group}",
    }


def validate_a1_definition(nodes: Iterable[Node], root_id: str) -> tuple[Node, ...]:
    supplied = tuple(nodes)
    indexed = {node.id: node for node in supplied}
    if len(indexed) != len(supplied):
        raise ValueError("Duplicate node identity in A1 input")
    ordered = reachable_nodes((root_id,), indexed.__getitem__)
    if {node.id for node in ordered} != set(indexed):
        raise ValueError("A1 input contains unreachable nodes")
    named = {n.name: n for n in ordered}
    if ordered[-1].name != ROOT or len(named) != len(ordered):
        raise ValueError("Wrong A1 root or conflicting shared identities")
    values = {}
    for group in (*_SOURCES, "definition"):
        name = f"neural/a1/{group}"
        stage = named.get(name)
        if stage is None or stage.kind != "composite" or stage.payload != _recipe(name):
            raise ValueError(f"Invalid A1 recipe: {name}")
        if {
            (e.role, e.slot): indexed[e.target_id].name for e in stage.dependencies
        } != _targets(group):
            raise ValueError(f"Invalid A1 slots: {name}")
        if group != "definition":
            config = named[f"neural/a1/config/{group}"]
            if (
                config.dependencies
                or config.payload["projection"] != f"a1-{group}"
                or config.payload["version"] != 1
            ):
                raise ValueError(f"Invalid A1 projection: {group}")
            values[group] = cast(dict[str, Any], config.to_dict()["payload"])["value"]
    _validate_values(values)
    return ordered


def build_a1_definition(
    config: Path,
    *,
    project_root: Path = PROJECT,
    model: str | None = None,
    rank: int | None = None,
    epochs: int | None = None,
    eval_batches: int | None = None,
    no_jit: bool = False,
) -> tuple[Node, ...]:
    captured: dict[Path, bytes] = {}

    def read(path: Path) -> bytes:
        if path not in captured:
            captured[path] = path.read_bytes()
        return captured[path]

    _, raw = read_yaml(config, read)
    for section in ("adapter", "training", "objective"):
        if section in raw and not isinstance(raw[section], dict):
            raise ValueError(f"Expected mapping for {section}")
    try:
        cfg = parse_configs(raw, config, A1Config)
        args = argparse.Namespace(
            model=model,
            rank=rank,
            epochs=epochs,
            eval_batches=eval_batches,
            no_jit=no_jit,
            output_dir=None,
            steps=None,
        )
        values = _project(apply_overrides(cfg, args))
    except (AttributeError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid A1 configuration: {error}") from error
    named: dict[str, Node] = {}
    for group, paths in _SOURCES.items():
        for path in paths:
            if path not in named:
                data = read(project_root / path)
                named[path] = Node(
                    kind="source",
                    name=path,
                    payload={
                        "sha256": sha256(data).hexdigest(),
                        "byte_length": len(data),
                    },
                )
        name = f"neural/a1/config/{group}"
        named[name] = Node(
            kind="semantic_content",
            name=name,
            payload={"projection": f"a1-{group}", "version": 1, "value": values[group]},
        )
    for group in (*_SOURCES, "definition"):
        name = f"neural/a1/{group}"
        named[name] = Node(
            kind="composite",
            name=name,
            payload=_recipe(name),
            dependencies=tuple(
                Dependency(role, slot, named[target].id)
                for (role, slot), target in _targets(group).items()
            ),
        )
    ordered = validate_a1_definition(named.values(), named[ROOT].id)
    for input_path, captured_bytes in captured.items():
        if input_path.read_bytes() != captured_bytes:
            raise ValueError(f"A1 input changed during capture: {input_path}")
    return ordered
