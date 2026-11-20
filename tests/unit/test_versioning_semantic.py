"""Offline registration tests using real semantic parser definitions."""

import subprocess
import sys
from pathlib import Path
from shutil import copytree

import pytest
import yaml
from pydantic import ValidationError

from parser.ontology.catalog import (
    DEFAULT_BUNDLE_MANIFEST,
    OntologyCatalogError,
    load_manifest_catalog,
)
from parser.semantic import schema
from versioning import Registry, export_manifest, hash_file, verify_manifest
from versioning.semantic import register_semantic_components

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    copytree(ROOT / "parser", root / "parser")
    copytree(ROOT / "configs/parser_config", root / "configs/parser_config")
    return root


def test_register_real_definitions_and_verify_manifest(tmp_path):
    registry = Registry(tmp_path / "registry")
    ids = register_semantic_components(registry, project_root=ROOT)
    assert register_semantic_components(registry, project_root=ROOT) == ids

    _, manifest = load_manifest_catalog()
    assert set(ids.ontology_modules) == set(manifest.modules)
    for name, record_id in ids.ontology_modules.items():
        record = registry.get(record_id)
        assert record.label == manifest.modules[name]
        assert record.label_role == "release"
        assert record.content_digest == hash_file(ROOT / f"parser/ontology/{name}.yaml")

    bundle = registry.get(ids.ontology_bundle)
    assert bundle.label == manifest.bundle_version
    assert bundle.content_digest == hash_file(DEFAULT_BUNDLE_MANIFEST)
    assert {dep.record_id for dep in bundle.dependencies} == set(
        ids.ontology_modules.values()
    )
    assert {dep.role for dep in bundle.dependencies} == {"ontology-module"}

    contract = registry.get(ids.contract)
    parser = registry.get(ids.parser)
    assert contract.label == schema.SEMANTIC_CONTRACT_VERSION
    assert contract.content_digest == hash_file(ROOT / "parser/semantic/schema.py")
    assert parser.kind == "parser-implementation"
    assert contract.kind == "parser-contract"
    assert contract.record_id != parser.record_id
    assert parser.details["source_sha256"]["parser/semantic/semantic_parser.py"] == (
        hash_file(ROOT / "parser/semantic/semantic_parser.py")
    )

    for name in ("router", "general", "discourse", "student"):
        prompt = registry.get(ids.prompts[name])
        assert prompt.label_role == "family"
        kind = "student-prompt" if name == "student" else "teacher-prompt"
        assert prompt.kind == kind
        inputs = {dep.record_id for dep in prompt.dependencies}
        expected_modules = {
            "router": set(),
            "general": set(ids.ontology_modules) - {"discourse"},
            "discourse": {"discourse"},
            "student": set(ids.ontology_modules),
        }
        assert inputs == {ids.contract} | {
            ids.ontology_modules[module] for module in expected_modules[name]
        }
    routed = registry.get(ids.prompts["routed"])
    assert {dep.record_id for dep in routed.dependencies} == {
        ids.contract,
        ids.prompts["general"],
        ids.prompts["discourse"],
    }

    data = export_manifest(registry, ids.parser)
    path = tmp_path / "portable.json"
    path.write_bytes(data)
    registry.root.rename(tmp_path / "moved-registry")
    verified = verify_manifest(path.read_bytes())
    assert verified[-1] == parser
    assert {record.record_id for record in verified} == {
        *ids.ontology_modules.values(),
        ids.ontology_bundle,
        ids.contract,
        *ids.prompts.values(),
        ids.parser,
    }


def test_ontology_byte_change_conflicts_with_existing_release(tmp_path, source_tree):
    registry = Registry(tmp_path / "registry")
    original = register_semantic_components(registry, project_root=source_tree)
    target = source_tree / "parser/ontology/general.yaml"
    target.write_bytes(target.read_bytes() + b"\n# changed bytes only\n")
    with pytest.raises(ValueError, match="Conflicting release label.*general"):
        register_semantic_components(registry, project_root=source_tree)
    record = registry.get(original.ontology_modules["general"])
    assert record.content_digest != hash_file(target)


@pytest.mark.parametrize("mutation", ["pin", "invalid-module"])
def test_registration_preserves_ontology_validation(tmp_path, source_tree, mutation):
    if mutation == "pin":
        target = source_tree / "parser/ontology/bundles" / DEFAULT_BUNDLE_MANIFEST.name
        payload = yaml.safe_load(target.read_text())
        payload["modules"]["general"] = "invalid-pin"
        message = "version mismatch"
    else:
        target = source_tree / "parser/ontology/general.yaml"
        payload = yaml.safe_load(target.read_text())
        payload["relations"] = {}
        message = "relations cannot be empty"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    registry = Registry(tmp_path / "registry")
    with pytest.raises(OntologyCatalogError, match=message):
        register_semantic_components(registry, project_root=source_tree)
    assert not registry.root.exists()


@pytest.mark.parametrize(
    ("path", "old", "new", "changed"),
    [
        (
            "configs/parser_config/parser_prompt.yaml",
            "routing component",
            "routing decision component",
            {"router"},
        ),
        (
            "configs/parser_config/general_prompt.yaml",
            "{catalog_guide}",
            "Catalog follows. {catalog_guide}",
            {"general", "routed"},
        ),
        (
            "configs/parser_config/discourse_prompt.yaml",
            "{discourse_catalog_guide}",
            "Discourse catalog follows. {discourse_catalog_guide}",
            {"discourse", "routed"},
        ),
        (
            "parser/semantic/catalog_prompt.py",
            "FINAL COMBINED EXTRACTION REQUIREMENTS",
            "COMBINED EXTRACTION REQUIREMENTS",
            {"router", "general", "discourse", "routed"},
        ),
        (
            "parser/student/student_prompt.py",
            "SOURCE TEXT",
            "DOCUMENT TEXT",
            {"student"},
        ),
    ],
)
def test_prompt_definition_change(tmp_path, source_tree, path, old, new, changed):
    registry = Registry(tmp_path / "registry")
    first = register_semantic_components(registry, project_root=source_tree)
    target = source_tree / path
    data = target.read_text()
    assert old in data
    target.write_text(data.replace(old, new, 1), encoding="utf-8")
    second = register_semantic_components(registry, project_root=source_tree)
    assert first.contract == second.contract
    assert first.ontology_bundle == second.ontology_bundle
    assert first.parser != second.parser
    assert {
        name for name in first.prompts if first.prompts[name] != second.prompts[name]
    } == changed
    for name in changed:
        before = registry.get(first.prompts[name])
        after = registry.get(second.prompts[name])
        assert before.content_digest != after.content_digest
        assert before.label == after.label


def test_parser_source_change_is_distinct_from_contract_change(tmp_path, source_tree):
    registry = Registry(tmp_path / "registry")
    first = register_semantic_components(registry, project_root=source_tree)
    target = source_tree / "parser/semantic/semantic_parser.py"
    target.write_bytes(target.read_bytes() + b"\n# implementation edit\n")
    second = register_semantic_components(registry, project_root=source_tree)
    assert first.contract == second.contract
    assert first.prompts == second.prompts
    first_digest = registry.get(first.parser).content_digest
    second_digest = registry.get(second.parser).content_digest
    assert first_digest != second_digest
    target = source_tree / "parser/semantic/schema.py"
    target.write_bytes(target.read_bytes() + b"\n# schema edit\n")
    third = register_semantic_components(registry, project_root=source_tree)
    assert third.contract != second.contract
    assert registry.get(third.contract).label == registry.get(second.contract).label
    assert registry.get(third.parser).content_digest == second_digest
    assert third.parser != second.parser


def test_existing_result_fields_and_strict_validation_are_preserved(tmp_path):
    register_semantic_components(Registry(tmp_path), project_root=ROOT)
    assert set(schema.SemanticParseResult.model_fields) == {
        "contract_version",
        "document",
        "provenance",
        "extraction",
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        schema.SemanticParseResult.model_validate({"version_record_id": "unexpected"})


def test_registration_avoids_model_and_provider_imports(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from versioning import Registry
from versioning.semantic import register_semantic_components

register_semantic_components(Registry(sys.argv[1]))
blocked = {"torch", "transformers", "jax", "numpy", "openai", "anthropic", "pydantic"}
assert not blocked.intersection(sys.modules)
""",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )


def test_git_provenance_is_optional_for_source_copies(tmp_path, source_tree):
    registry = Registry(tmp_path / "registry")
    plain = register_semantic_components(registry, project_root=ROOT)
    copied = register_semantic_components(
        registry, project_root=source_tree, include_git_provenance=True
    )
    assert copied == plain
    assert registry.get(copied.parser).producer is None
