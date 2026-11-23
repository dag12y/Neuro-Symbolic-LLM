"""Exercise the actual student trainer and checkpoint loader with a tiny local model."""

import json
from pathlib import Path

import pytest
import yaml

from tests.semantic_fixtures import pair, result
from versioning.__main__ import main
from versioning.lineage import link
from versioning.manifest import export_manifest, verify_manifest
from versioning.registry import Registry


def test_local_student_model_training_and_evaluation(tmp_path, monkeypatch, capsys):
    pytest.importorskip("accelerate")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from parser.semantic.semantic_parser import DistilledSemanticParser

    model = tmp_path / "base"
    model.mkdir()
    GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=32,
            vocab_size=4,
            eos_token_id=1,
            bos_token_id=0,
        )
    ).save_pretrained(model)
    tok = Tokenizer(
        WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "test": 3}, unk_token="[UNK]")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    ).save_pretrained(model)
    config = yaml.safe_load(
        Path("configs/parser_config/student_config.yaml").read_bytes()
    )
    config["student"]["base_model"] = str(model)
    config["training"].update(epochs=1, batch_size=1, gradient_accumulation_steps=1)
    config["data"]["val_split"] = 0
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(config))
    pairs = tmp_path / "pairs"
    pairs.write_text(json.dumps(pair()) + "\n")
    output = tmp_path / "student"
    db = tmp_path / "graph.db"
    reg = Registry(db)

    def run(*args):
        assert main([str(x) for x in (*args, "--registry", db)]) == 0
        return capsys.readouterr().out.strip().splitlines()[-1]

    trained = run(
        "train-student", "--train-file", pairs, "--config", cfg, "--output-dir", output
    )
    assert (output / "model.safetensors").is_file()
    sentences = tmp_path / "sentences"
    sentences.write_text("Mira left.\n")
    # Load the saved model and tokenizer; replace generation to avoid a long decode.
    monkeypatch.setattr(
        DistilledSemanticParser, "parse", lambda self, text: result(text)
    )
    evaluated = run(
        "evaluate-student",
        "--model-dir",
        output,
        "--sentences",
        sentences,
        "--output-dir",
        tmp_path / "eval",
    )
    root = link(reg, evaluated, {"model/" + p.name: trained for p in output.iterdir()})
    graph = verify_manifest(export_manifest(reg, root.id))
    assert reg.get(trained) in graph
    assert "Validated documents: 1/1" in (tmp_path / "eval/evaluation.txt").read_text()
