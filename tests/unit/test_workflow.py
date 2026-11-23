"""Cross-stage CLI fixtures. The Git package below is a test double for the five public CLIs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from parser.semantic import model_config
from parser.semantic.backends import CallableBackend
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParserConfig,
)
from tests.semantic_fixtures import target
from versioning.__main__ import main
from versioning.lineage import link, verify_file
from versioning.manifest import export_manifest, verify_manifest
from versioning.mining import release
from versioning.registry import Registry
from versioning.workflow import composite


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def fixture_repo(path: Path) -> Path:
    path.mkdir()
    git(path, "init", "-q")
    git(
        path,
        "remote",
        "add",
        "origin",
        "https://github.com/iCog-Labs-Dev/hybrid-miner.git",
    )
    pkg = path / "src/hybrid_miner"
    (pkg / "mining").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "mining/__init__.py").write_text("")
    # Only the input/output option contract is simulated. No mining algorithm.
    script = """import sys,json
from pathlib import Path
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
for k,v in args.items():
    if k.endswith('output'):
        Path(v).write_bytes(b'')
if '--accepted-output' in args:
    Path(args['--accepted-output']).write_bytes(Path(args['--input']).read_bytes())
if '--exact' in args:
    d=json.loads(Path(args['--documents']).read_text().splitlines()[0])
    pat={'id':'fixture','contract_version':'pattern-record-v1','provenance':{'corpus_id':d['corpus_id'],'source_document_ids':[d['document_id']]},'source_spans':[{'document_id':d['document_id'],'start':0,'end':4}]}
    Path(args['--output']).write_text(json.dumps(pat)+'\\n')
"""
    for module in (
        "ingestion",
        "emission",
        "mining/admitted_support",
        "mining/exact_matches",
        "mining/build_pattern_records",
    ):
        (pkg / f"{module}.py").write_text(script)
    git(path, "add", "src")
    git(
        path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "synthetic CLI contract",
    )
    git(path, "tag", "v0.1.1")
    return path


@pytest.fixture
def cli(tmp_path, capsys):
    db = tmp_path / "registry.db"

    def run(*args, code=0):
        assert main([str(a) for a in (*args, "--registry", db)]) == code
        return capsys.readouterr().out.strip().splitlines()[-1]

    return run, Registry(db)


def test_parser_miner_embedding_chain(tmp_path, cli, monkeypatch):
    pytest.importorskip("hybrid_miner")
    encoder = pytest.importorskip("offline.embedding.run")
    run, reg = cli
    text = "Mira left."
    source = tmp_path / "text.jsonl"
    source.write_text(
        json.dumps({"document_id": "one", "corpus_id": "fixture", "source_text": text})
        + "\ninvalid\n"
    )

    def factory(config, *, catalog, require_structured_output):
        return ReferenceSemanticParser(
            backend=CallableBackend(lambda **_: json.dumps(target(text))),
            config=SemanticParserConfig(model_name="fixture", enable_routing=False),
            catalog=catalog,
        )

    monkeypatch.setattr(model_config, "build_reference_semantic_parser", factory)
    p = tmp_path / "parsed"
    parser = run(
        "run-parser-export",
        "--observe-runtime",
        "--input",
        source,
        "--output-dir",
        p,
        code=2,
    )
    d = tmp_path / "docs"
    converted = run(
        "run-symbolic-export",
        "--input",
        p / "parser-results.jsonl",
        "--bundle-output",
        "--output-dir",
        d,
    )
    documents = link(reg, converted, {"input": parser})
    repo = fixture_repo(tmp_path / "fixture-source")
    m = tmp_path / "mined"
    args = [
        "run-miner",
        "--source",
        repo,
        "--petta-root",
        tmp_path,
        "--miner-root",
        tmp_path,
        "--bundle",
        d / "ontology-bundle.json",
        "--documents",
        d / "documents.jsonl",
    ]
    mined = run(*args, "--output-dir", m)
    mining = link(
        reg, mined, {"bundle.json": documents.id, "documents.jsonl": documents.id}
    )

    class Encoder:
        pooling = "fixture"
        layer_idx = 1

        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.substrate = SimpleNamespace(
                tokenizer=None,
                architecture=SimpleNamespace(num_layers=2, hidden_size=3),
            )

        def encode(self, texts):
            return np.ones((len(texts), 3), np.float32)

    monkeypatch.setattr(encoder, "FrozenLLMEncoder", Encoder)
    e = tmp_path / "emb"
    embedded = run(
        "run-embedding",
        "--patterns",
        m / "patterns.jsonl",
        "--documents",
        m / "accepted.jsonl",
        "--output-dir",
        e,
        "--model",
        "fixture",
        "--symbolic-dim",
        "2",
    )
    final = link(
        reg, embedded, {"patterns.jsonl": mining.id, "documents.jsonl": mining.id}
    )
    nodes = verify_manifest(export_manifest(reg, final.id), expected_root_id=final.id)
    assert reg.get(parser) in nodes
    output = next(
        n for n in nodes if n.name == "workflow/embedding/output/embeddings.jsonl"
    )
    verify_file(nodes, output.id, e / "embeddings.jsonl")
    (e / "embeddings.jsonl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        verify_file(nodes, output.id, e / "embeddings.jsonl")
    with pytest.raises(ValueError, match="Incompatible"):
        link(reg, embedded, {"patterns.jsonl": documents.id})
    forged = composite(
        "workflow/linked-run",
        "run",
        {"consumer": reg.get(embedded), "patterns.jsonl": documents},
    )
    reg.register_batch([forged])
    with pytest.raises(ValueError, match="Incompatible"):
        verify_manifest(export_manifest(reg, forged.id))
    # Dirty or installed code cannot replace the selected immutable source.
    (repo / "src/hybrid_miner/ingestion.py").write_text("raise RuntimeError()")
    repeated = run(*args, "--output-dir", tmp_path / "repeat")
    assert repeated == mined
    git(repo, "add", "src")
    git(
        repo,
        "-c",
        "user.name=F",
        "-c",
        "user.email=f@example.invalid",
        "commit",
        "-qm",
        "failed fixture",
    )
    git(repo, "tag", "-f", "v0.1.1")
    failed = run(*args, "--output-dir", tmp_path / "failure", code=1)
    assert failed != mined
    graph = verify_manifest(export_manifest(reg, failed))
    assert all(n.kind != "artifact" for n in graph)


def test_missing_miner_and_independent_import(cli, tmp_path):
    run, reg = cli
    source = tmp_path / "empty"
    source.write_bytes(b"")
    imported = reg.get(run("import-file", source, "--kind", "patterns"))
    assert imported.kind == "source" and not imported.dependencies
    failed = run(
        "run-miner",
        "--documents",
        source,
        "--bundle",
        source,
        "--petta-root",
        tmp_path,
        "--miner-root",
        tmp_path,
        "--output-dir",
        tmp_path / "out",
        code=1,
    )
    graph = verify_manifest(export_manifest(reg, failed))
    assert not any(n.kind == "artifact" for n in graph)
    assert release(None)[1].payload["value"]["commit"] is None


def test_parser_failure(cli, tmp_path, monkeypatch):
    run, reg = cli

    def fail(*args, **kwargs):
        raise RuntimeError("fixture")

    monkeypatch.setattr(model_config, "build_reference_semantic_parser", fail)
    source = tmp_path / "source.txt"
    source.write_text("source")
    root = run(
        "run-parser-export",
        "--observe-runtime",
        "--input",
        source,
        "--output-dir",
        tmp_path / "out",
        code=1,
    )
    nodes = verify_manifest(export_manifest(reg, root))
    assert nodes[-1].name == "workflow/parser/failed"
    assert not any(n.kind == "artifact" for n in nodes)


def test_a1_observed_train_and_independent_evaluation(tmp_path, cli, monkeypatch):
    pytest.importorskip("optax")
    pytest.importorskip("torchax")
    import yaml

    from experiments import run_stage_a1
    from stages import stage_base
    from tests.frozenllm.conftest import make_substrate

    run, reg = cli

    class Tokenizer:
        def __call__(self, text, return_tensors):
            return {
                "input_ids": np.asarray([[ord(c) % 63 for c in text]], dtype=np.int32)
            }

    _, sub = make_substrate("gpt2")
    sub._tokenizer = Tokenizer()
    monkeypatch.setattr(run_stage_a1, "FrozenSubstrate", lambda name: sub)
    monkeypatch.setattr(
        stage_base, "load_dataset", lambda *args, **kwargs: {"text": ["Test", "rows"]}
    )
    cfg = tmp_path / "a1.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "adapter": {"rank": 2, "layers": [8]},
                "training": {
                    "model": "fixture",
                    "seq_len": 4,
                    "batch_size": 2,
                    "epochs": 1,
                    "eval_batches": 1,
                },
                "domains": [{"name": "demo", "dataset": "fixture"}],
            }
        )
    )
    out = tmp_path / "trained"
    trained = run("run-a1", "--config", cfg, "--output-dir", out)
    archive = out / "checkpoints/task_0_demo.npz"
    evaluated = run(
        "evaluate-a1",
        "--config",
        cfg,
        "--checkpoint",
        archive,
        "--output-dir",
        tmp_path / "eval",
    )
    root = link(reg, evaluated, {"checkpoint.npz": trained})
    nodes = verify_manifest(export_manifest(reg, root.id))
    assert any(n.name == "workflow/a1-eval/output/evaluation.json" for n in nodes)
    observation = next(n for n in nodes if n.name == "workflow/a1-train/runtime")
    assert [d["split"] for d in observation.payload["value"]["datasets"]] == [
        "train",
        "validation",
        "test",
    ]
    monkeypatch.setattr(
        stage_base, "load_dataset", lambda *args, **kwargs: {"text": []}
    )
    for cmd, extra in [("run-a1", []), ("evaluate-a1", ["--checkpoint", archive])]:
        root_id = run(
            cmd, "--config", cfg, *extra, "--output-dir", tmp_path / cmd, code=1
        )
        assert not any(
            n.kind == "artifact" for n in verify_manifest(export_manifest(reg, root_id))
        )


def test_interrupted_workflow_publication(cli, tmp_path, monkeypatch):
    from versioning import workflow
    from versioning.workflow_stages import stage_definition

    _, reg = cli
    names = {"documents.jsonl", "patterns.jsonl"}
    files = {n: tmp_path / n for n in names}
    for path in files.values():
        path.write_bytes(b"observed")

    def producer(private, out, runtime):
        runtime.update(
            versions={"python": "fixture"},
            substrate={
                "model_revision_reported": None,
                "tokenizer_revision_reported": None,
                "tokenizer_class": "Fixture",
                "layers": 1,
                "hidden_size": 1,
            },
        )
        (out / "embeddings.jsonl").write_bytes(b"fixture")

    def interrupted(*args, **kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(workflow.os, "link", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        workflow.capture(
            "embedding",
            reg,
            files,
            tmp_path / "out",
            lambda: stage_definition("embedding"),
            {
                "model": "fixture",
                "max_length": 1,
                "layer_idx": 0,
                "symbolic_dim": 1,
                "projection_seed": 1,
            },
            producer,
        )
    assert not reg.path.exists()


def test_installed_package_comparison_and_tag_resolution(tmp_path, monkeypatch):
    import importlib.metadata

    from versioning.mining import installed_release

    repo = fixture_repo(tmp_path / "source")
    # Return a distribution location with a version that resembles the tag.
    # The comparison must still use the exact materialized package bytes.
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(
            version="0.1.1", locate_file=lambda name: repo / "src" / name
        ),
    )
    matched = installed_release(repo)
    assert matched[-2].payload["value"]["comparison"] == "match"
    (repo / "src/hybrid_miner/emission.py").write_text("changed")
    changed = installed_release(repo)
    assert changed[-2].payload["value"]["comparison"] == "mismatch"
    assert changed[0] == matched[0] and changed[1] == matched[1]


def test_incompatible_patterns_fail_without_producer(cli, tmp_path):
    from versioning.workflow import slots

    run, reg = cli
    doc = tmp_path / "doc.jsonl"
    doc.write_text('{"contract_version":"wrong"}\n')
    failed = run(
        "run-embedding",
        "--documents",
        doc,
        "--patterns",
        doc,
        "--output-dir",
        tmp_path / "out",
        code=1,
    )
    nodes = verify_manifest(export_manifest(reg, failed))
    outcome = slots(nodes[-1], {n.id: n for n in nodes})["outcome"]
    assert outcome.payload["value"]["status"] == "failed"
    assert not any(n.kind == "artifact" for n in nodes)
