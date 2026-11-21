"""Dependency-light configuration used by the Stage A1 driver and tracker."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

ACTIVATION_NAMES = ("identity", "gelu", "relu", "tanh")
SECTIONS = ("adapter", "objective", "training", "domains")
# Domain names become file names (params/task_<t>_<name>.npz).
DOMAIN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def checked_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {unknown}. Allowed: {sorted(known)}."
        )
    return dict(data)


@dataclass(frozen=True)
class ResidualConfig:
    """Adapter hyper-parameters; mirrors the ``adapter:`` YAML section."""

    rank: int = 16
    activation: str = "identity"
    alpha: float | None = None
    init_std: float | None = None
    late_start: int | None = None
    late_end: int | None = None
    # Explicit blocks; replaces the late range, exclusive with late_start/late_end.
    layers: list[int] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be an integer >= 1, got {self.rank!r}")
        if self.activation not in ACTIVATION_NAMES:
            raise ValueError(
                f"Unknown activation {self.activation!r}. "
                f"Expected one of {sorted(ACTIVATION_NAMES)}."
            )
        if self.alpha is not None and not self.alpha > 0:
            raise ValueError(f"alpha must be > 0 or null, got {self.alpha!r}")
        if self.init_std is not None and not self.init_std > 0:
            raise ValueError(f"init_std must be > 0 or null, got {self.init_std!r}")
        for name in ("late_start", "late_end"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer or null, got {value!r}")
        if self.layers is not None:
            if (
                not isinstance(self.layers, list | tuple)
                or not self.layers
                or not all(
                    isinstance(i, int) and not isinstance(i, bool) for i in self.layers
                )
            ):
                raise ValueError(
                    f"layers must be a non-empty list of integers or null, "
                    f"got {self.layers!r}"
                )
            if self.late_start is not None or self.late_end is not None:
                raise ValueError(
                    "Set either layers or late_start/late_end, not both "
                    f"(layers={self.layers!r}, late_start={self.late_start!r}, "
                    f"late_end={self.late_end!r})."
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResidualConfig:
        return cls(**checked_kwargs(cls, data))

    @property
    def scale(self) -> float:
        return 1.0 if self.alpha is None else float(self.alpha) / self.rank


@dataclass(frozen=True)
class A1Config:
    lambda_kl: float = 0.0
    lambda_wd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("lambda_kl", "lambda_wd"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or value < 0:
                raise ValueError(f"{name} must be a number >= 0, got {value!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A1Config:
        return cls(**checked_kwargs(cls, data))


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "gpt2"
    seq_len: int = 128
    batch_size: int = 8
    # Default epochs per domain; a domain's ``epochs`` overrides it.
    epochs: int = 3
    learning_rate: float = 1.0e-3
    eval_batches: int = 8
    log_every: int = 10
    seed: int = 0
    # Fresh Adam moments at each domain boundary.
    reset_optimizer: bool = True
    output_dir: str = "runs"
    jit: bool = True

    def __post_init__(self) -> None:
        for name in ("seq_len", "batch_size", "eval_batches", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {self.epochs}")
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not isinstance(self.jit, bool):
            raise ValueError(f"jit must be a boolean, got {self.jit!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        return cls(**checked_kwargs(cls, data))


@dataclass(frozen=True)
class DomainConfig:
    name: str
    dataset: str
    dataset_config: str | None = None
    text_field: str = "text"
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    # Joins rows into one stream; use "\n\n" for one-document-per-row datasets.
    separator: str = ""
    epochs: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not DOMAIN_NAME.match(self.name):
            raise ValueError(
                f"Domain name must match {DOMAIN_NAME.pattern}, got {self.name!r}"
            )
        if self.epochs is not None and (
            not isinstance(self.epochs, int) or self.epochs < 0
        ):
            raise ValueError(f"Domain {self.name!r}: epochs must be >= 0 or null.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DomainConfig:
        return cls(**checked_kwargs(cls, data))


@dataclass(frozen=True)
class RunConfig:
    adapter: ResidualConfig
    # Stage objective config: any frozen dataclass with ``from_dict``.
    objective: Any
    training: TrainingConfig
    domains: tuple[DomainConfig, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": asdict(self.adapter),
            "objective": asdict(self.objective),
            "training": asdict(self.training),
            "domains": [asdict(d) for d in self.domains],
        }


def load_configs(path: Path, objective_cls: type[Any]) -> RunConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    return parse_configs(raw, path, objective_cls)


def parse_configs(
    raw: Mapping[str, Any], path: Path, objective_cls: type[Any]
) -> RunConfig:
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown config sections in {path}: {unknown}.")
    domains_raw = raw.get("domains")
    if not isinstance(domains_raw, list) or not domains_raw:
        raise ValueError(f"{path}: 'domains' must be a non-empty list of domains.")
    domains = tuple(DomainConfig.from_dict(d) for d in domains_raw)
    names = [d.name for d in domains]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: domain names must be unique, got {names}.")
    return RunConfig(
        adapter=ResidualConfig.from_dict(raw.get("adapter") or {}),
        objective=objective_cls.from_dict(raw.get("objective") or {}),
        training=TrainingConfig.from_dict(raw.get("training") or {}),
        domains=domains,
    )


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    adapter, training, domains = cfg.adapter, cfg.training, cfg.domains
    if args.rank is not None:
        adapter = replace(adapter, rank=args.rank)
    if args.model is not None:
        training = replace(training, model=args.model)
    if args.output_dir is not None:
        training = replace(training, output_dir=args.output_dir)
    if getattr(args, "no_jit", False):
        training = replace(training, jit=False)
    if args.epochs is not None and args.steps is not None:
        raise ValueError("Cannot specify both --epochs and --steps. Use --epochs.")
    if args.steps is not None:
        raise ValueError("--steps has been replaced by --epochs. Use --epochs.")
    if args.epochs is not None:
        if args.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {args.epochs}")
        training = replace(training, epochs=args.epochs)
        domains = tuple(replace(d, epochs=args.epochs) for d in domains)
    if getattr(args, "eval_batches", None) is not None:
        if args.eval_batches < 1:
            raise ValueError(f"eval_batches must be >= 1, got {args.eval_batches}")
        training = replace(training, eval_batches=args.eval_batches)
    return replace(cfg, adapter=adapter, training=training, domains=domains)
