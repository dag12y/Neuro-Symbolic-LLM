from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from stages.config import A1Config, apply_overrides, load_configs
from versioning import neural
from versioning.__main__ import main
from versioning.manifest import export_manifest, verify_manifest
from versioning.nodes import Dependency, Node
from versioning.registry import Registry


def values(nodes: tuple[Node, ...]) -> dict[str, Any]:
    return {
        n.name.split("/")[-1]: cast(dict[str, Any], n.to_dict()["payload"])["value"]
        for n in nodes
        if n.kind == "semantic_content"
    }


@pytest.mark.parametrize("override", [False, True])
def test_effective_runtime_parity(tmp_path: Path, override: bool) -> None:
    config = tmp_path / "config.yaml"
    raw = yaml.safe_load(neural.DEFAULT_CONFIG.read_text())
    raw["domains"][0]["epochs"] = 0
    raw["adapter"]["seed"] = 19
    raw["training"]["seed"] = 42
    config.write_text(yaml.safe_dump(raw))
    options: dict[str, Any] = (
        {
            "model": "local/model",
            "epochs": 2,
            "rank": 7,
            "no_jit": True,
            "eval_batches": 3,
        }
        if override
        else {}
    )
    args = argparse.Namespace(
        **(
            {
                "model": None,
                "rank": None,
                "no_jit": False,
                "eval_batches": None,
                "epochs": None,
            }
            | options
        ),
        steps=None,
        output_dir="/elsewhere",
        run_name="ignored",
    )
    cfg = apply_overrides(load_configs(config, A1Config), args)
    got = values(neural.build_a1_definition(config, **options))
    assert got["substrate"] == {
        "model": cfg.training.model,
        "tokenizer": cfg.training.model,
    }
    assert got["adapter"]["rank"] == cfg.adapter.rank
    assert got["adapter"]["seed"] == cfg.adapter.seed == 19
    opt = got["optimization"]
    assert opt["seed"] == cfg.training.seed == 42
    assert (
        opt["optimizer"] == "adam"
        and opt["reset_optimizer"] == cfg.training.reset_optimizer
    )
    for key in ("lambda_wd", "lambda_kl", "learning_rate"):
        assert struct.unpack(">d", bytes.fromhex(opt[key]))[0] == getattr(
            cfg.objective, key, cfg.training.learning_rate
        )
    assert opt["epochs"] == [
        cfg.training.epochs if d.epochs is None else d.epochs for d in cfg.domains
    ]
    assert [d["name"] for d in got["data"]["domains"]] == [d.name for d in cfg.domains]
    assert got["evaluation"]["eval_batches"] == cfg.training.eval_batches
    assert got["driver"]["jit"] == cfg.training.jit


def test_defaults_comments_and_destinations(tmp_path: Path) -> None:
    p = tmp_path / "a.yaml"
    p.write_text("domains: [{name: one, dataset: data}]")
    first = neural.build_a1_definition(p)
    assert values(first)["data"]["batch_size"] == 8
    raw = load_configs(p, A1Config).to_dict()
    raw["training"]["output_dir"] = "/some/other/path"
    p.write_text("# formatting\n" + yaml.safe_dump(raw))
    assert neural.build_a1_definition(p) == first


@pytest.mark.parametrize(
    "mutation", ["domain-order", "dataset", "objective", "layers", "rank"]
)
def test_meaningful_changes(tmp_path: Path, mutation: str) -> None:
    raw = yaml.safe_load(neural.DEFAULT_CONFIG.read_text())
    if mutation == "domain-order":
        raw["domains"].reverse()
    elif mutation == "dataset":
        raw["domains"][0]["train_split"] = "train[:2]"
    elif mutation == "objective":
        raw["objective"]["lambda_wd"] = 0.2
    elif mutation == "layers":
        raw["adapter"]["layers"] = [1, 3]
    else:
        raw["adapter"]["rank"] = 32
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert (
        neural.build_a1_definition(p)[-1]
        != neural.build_a1_definition(neural.DEFAULT_CONFIG)[-1]
    )


@pytest.mark.parametrize(
    "extra",
    [
        "adapter: {rank: true}",
        "objective: {lambda_kl: .nan}",
        "training: {model: /host/model}",
        "adapter: {layers: [1, 1]}",
        "training: {seed: 1, seed: 2}",
        "adapter: []",
    ],
)
def test_invalid_config(tmp_path: Path, extra: str) -> None:
    p = tmp_path / "a.yaml"
    p.write_text("domains: [{name: one, dataset: data}]\n" + extra)
    with pytest.raises(ValueError):
        neural.build_a1_definition(p)


def test_source_change_and_capture_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = neural.build_a1_definition(neural.DEFAULT_CONFIG)
    for node in first:
        if node.kind == "source":
            dest = tmp_path / node.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(neural.PROJECT / node.name, dest)
    path = tmp_path / "residual/adapter.py"
    path.write_bytes(path.read_bytes() + b"\n")
    second = neural.build_a1_definition(neural.DEFAULT_CONFIG, project_root=tmp_path)
    assert first[-1] != second[-1]
    old = Path.read_bytes

    def changed(p: Path) -> bytes:
        data = old(p)
        if p == path:
            p.write_bytes(data + b"\n")
        return data

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="changed during capture"):
        neural.build_a1_definition(neural.DEFAULT_CONFIG, project_root=tmp_path)


@pytest.mark.parametrize(
    "case", ["slot", "projection", "version", "unexpected", "duplicate-domain"]
)
def test_rehashed_graph(tmp_path: Path, case: str) -> None:
    original = neural.build_a1_definition(neural.DEFAULT_CONFIG)
    changed: dict[str, Node] = {}
    for old in original:
        deps = tuple(
            Dependency(e.role, e.slot, changed[e.target_id].id)
            for e in old.dependencies
        )
        p = cast(dict[str, Any], old.to_dict()["payload"])
        if old.name == "neural/a1/config/data":
            if case == "projection":
                p["value"]["seq_len"] = True
            if case == "duplicate-domain":
                p["value"]["domains"][1]["name"] = p["value"]["domains"][0]["name"]
            if case == "unexpected":
                src = next(n for n in changed.values() if n.kind == "source")
                deps = (Dependency("uses-source", "unexpected", src.id),)
        if old.name == neural.ROOT:
            if case == "version":
                p["recipe"]["version"] = 2
            if case == "slot":
                deps = (Dependency("uses-component", "unknown", deps[0].target_id),)
        changed[old.id] = replace(old, payload=p, dependencies=deps)
    reg = Registry(tmp_path / "g")
    reg.register_batch(changed.values())
    root = changed[original[-1].id].id
    verified = verify_manifest(export_manifest(reg, root))
    with pytest.raises(ValueError):
        neural.validate_a1_definition(verified, root)


def test_cli_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "g"
    args = ["snapshot-a1", "--registry", str(db), "--rank", "4"]
    for _ in range(2):
        assert main(args) == 0
        root = capsys.readouterr().out.strip()
        assert root == neural.build_a1_definition(neural.DEFAULT_CONFIG, rank=4)[-1].id
    data = export_manifest(Registry(db), root)
    db.unlink()
    assert neural.validate_a1_definition(verify_manifest(data), root)[-1].id == root
    with pytest.raises(SystemExit):
        main([*args, "--steps", "5"])
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from versioning.neural import *; import sys; build_a1_definition(DEFAULT_CONFIG); assert not set(sys.modules)&{'torch','torchax','jax','transformers','datasets'}",
        ],
        check=True,
    )


def test_runtime_entrypoint_parity_when_installed() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("torchax")
    pytest.importorskip("optax")
    from experiments import run_stage_a1 as runner

    args = runner.parse_args(
        ["--rank", "12", "--epochs", "1", "--run-name", "temporary"]
    )
    runtime = runner.apply_overrides(
        runner.load_configs(args.config, runner.A1Config), args
    )
    assert neural._project(runtime) == values(
        neural.build_a1_definition(args.config, rank=12, epochs=1)
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing-source",
        "unexpected-source",
        "wrong-source-name",
        "wrong-projection-version",
        "wrong-component",
        "wrong-settings",
        "shared-source",
    ],
)
def test_rehashed_inventory_and_relationship_attacks(tmp_path: Path, case: str) -> None:
    original = neural.build_a1_definition(neural.DEFAULT_CONFIG)
    by_name = {node.name: node for node in original}
    replacements: dict[str, Node] = {}
    for old in original:
        payload = cast(dict[str, Any], old.to_dict()["payload"])
        dependencies = tuple(
            Dependency(edge.role, edge.slot, replacements[edge.target_id].id)
            for edge in old.dependencies
        )
        name = old.name
        if (
            case in {"missing-source", "unexpected-source", "wrong-source-name"}
            and name == "neural/a1/substrate"
        ):
            changed_dependencies = list(dependencies)
            if case == "missing-source":
                changed_dependencies = [
                    edge
                    for edge in changed_dependencies
                    if edge.slot != "frozenllm/substrate/loader.py"
                ]
            elif case == "unexpected-source":
                source = next(
                    node for node in replacements.values() if node.kind == "source"
                )
                changed_dependencies.append(
                    Dependency("uses-source", "extra", source.id)
                )
            else:
                changed_dependencies = [
                    Dependency(
                        edge.role,
                        edge.slot,
                        (
                            by_name["frozenllm/substrate/architecture.py"].id
                            if edge.slot == "frozenllm/substrate/loader.py"
                            else edge.target_id
                        ),
                    )
                    for edge in changed_dependencies
                ]
            dependencies = tuple(changed_dependencies)
        if case == "wrong-projection-version" and name == "neural/a1/config/data":
            payload["version"] = 2
        if case == "wrong-component" and name == "neural/a1/driver":
            dependencies = tuple(
                Dependency(
                    edge.role,
                    edge.slot,
                    (
                        by_name["neural/a1/substrate"].id
                        if edge.slot == "optimization"
                        else edge.target_id
                    ),
                )
                for edge in dependencies
            )
        if case == "wrong-settings" and name == "neural/a1/config/optimization":
            payload["value"]["optimizer"] = "sgd"
        if case == "shared-source" and name == "neural/a1/optimization":
            dependencies = tuple(
                Dependency(
                    edge.role,
                    edge.slot,
                    (
                        by_name["stages/stage_base.py"].id
                        if edge.slot == "stages/config.py"
                        else edge.target_id
                    ),
                )
                for edge in dependencies
            )
        replacements[old.id] = replace(
            old, name=name, payload=payload, dependencies=tuple(dependencies)
        )
    reg = Registry(tmp_path / f"{case}.sqlite3")
    reg.register_batch(replacements.values())
    root = replacements[original[-1].id].id
    graph = verify_manifest(export_manifest(reg, root))
    with pytest.raises((ValueError, KeyError)):
        neural.validate_a1_definition(graph, root)


def test_validator_rejects_unreachable_supplied_nodes() -> None:
    graph = neural.build_a1_definition(neural.DEFAULT_CONFIG)
    orphan = Node(
        kind="source", name="orphan.py", payload={"sha256": "0" * 64, "byte_length": 0}
    )
    with pytest.raises(ValueError, match="unreachable"):
        neural.validate_a1_definition((*graph, orphan), graph[-1].id)


def test_runtime_config_aliases_are_shared_when_runtime_imports() -> None:
    pytest.importorskip("jax")
    from experiments import run_stage_a1
    from residual import adapter
    from stages import config, stage_base
    from stages.stage_A import a1

    assert run_stage_a1.apply_overrides is config.apply_overrides
    assert stage_base.load_configs is config.load_configs
    assert adapter.ResidualConfig is config.ResidualConfig
    assert a1.A1Config is config.A1Config


def test_explicit_defaults_equal_omitted_defaults(tmp_path: Path) -> None:
    raw = yaml.safe_load(neural.DEFAULT_CONFIG.read_text())
    omitted = tmp_path / "omitted.yaml"
    explicit = tmp_path / "explicit.yaml"
    omitted.write_text("domains: [{name: one, dataset: demo}]\n")
    raw = {
        "domains": [{"name": "one", "dataset": "demo"}],
        "adapter": {
            "rank": 16,
            "activation": "identity",
            "alpha": None,
            "init_std": None,
            "late_start": None,
            "late_end": None,
            "layers": None,
            "seed": 0,
        },
        "objective": {"lambda_kl": 0.0, "lambda_wd": 0.0},
        "training": {
            "model": "gpt2",
            "seq_len": 128,
            "batch_size": 8,
            "epochs": 3,
            "learning_rate": 0.001,
            "eval_batches": 8,
            "log_every": 10,
            "seed": 0,
            "reset_optimizer": True,
            "output_dir": "elsewhere",
            "jit": True,
        },
    }
    explicit.write_text(yaml.safe_dump(raw))
    assert neural.build_a1_definition(omitted) == neural.build_a1_definition(explicit)


def test_rehashed_duplicate_shared_source_identity(tmp_path: Path) -> None:
    original = neural.build_a1_definition(neural.DEFAULT_CONFIG)
    duplicate = Node(
        kind="source",
        name="stages/config.py",
        payload={"sha256": "0" * 64, "byte_length": 1},
    )
    changed: dict[str, Node] = {}
    for old in original:
        dependencies = tuple(
            Dependency(
                edge.role,
                edge.slot,
                (
                    duplicate.id
                    if old.name == "neural/a1/adapter"
                    and edge.slot == "stages/config.py"
                    else changed[edge.target_id].id
                ),
            )
            for edge in old.dependencies
        )
        changed[old.id] = Node(
            kind=old.kind,
            name=old.name,
            payload=cast(dict[str, Any], old.to_dict()["payload"]),
            dependencies=dependencies,
        )
    graph = (*changed.values(), duplicate)
    reg = Registry(tmp_path / "duplicate-source.sqlite3")
    reg.register_batch(graph)
    root = changed[original[-1].id].id
    verified = verify_manifest(export_manifest(reg, root))
    with pytest.raises(ValueError, match="conflicting shared identities"):
        neural.validate_a1_definition(verified, root)


def test_checked_in_a1_configuration_matches_shared_runtime_loader() -> None:
    args = argparse.Namespace(
        model=None,
        rank=None,
        epochs=None,
        steps=None,
        no_jit=False,
        eval_batches=None,
        output_dir="/temporary/output",
        run_name="omitted",
    )
    runtime = apply_overrides(load_configs(neural.DEFAULT_CONFIG, A1Config), args)
    assert neural._project(runtime) == values(
        neural.build_a1_definition(neural.DEFAULT_CONFIG)
    )
    assert runtime.training.batch_size == 16
    assert [domain.name for domain in runtime.domains] == ["eurlex", "ledgar", "scotus"]
    assert [domain.epochs for domain in runtime.domains] == [None, None, None]
