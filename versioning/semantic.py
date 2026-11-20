"""Register a source tree's semantic parser definitions without loading models."""

from __future__ import annotations

import ast
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

from parser.ontology.catalog import DEFAULT_BUNDLE_MANIFEST, load_manifest_catalog
from parser.semantic.catalog_prompt import _load_prompt_config

from .hashing import canonical_json
from .records import ComponentRecord, Dependency
from .registry import Registry

_SCHEMA = "parser/semantic/schema.py"
_PARSER = "parser/semantic/semantic_parser.py"
_TEACHER = "parser/semantic/catalog_prompt.py"
_STUDENT = "parser/student/student_prompt.py"
_PARSER_SOURCES = (
    _PARSER,
    "parser/grammar/atomese.py",
    "parser/ontology/catalog.py",
    "parser/semantic/backends.py",
    "parser/semantic/metta_renderer.py",
    "parser/semantic/normalization.py",
    "parser/semantic/output_cleanup.py",
)
_PROMPT_FILES = {
    "router": "parser_prompt.yaml",
    "general": "general_prompt.yaml",
    "discourse": "discourse_prompt.yaml",
}


@dataclass(frozen=True, slots=True)
class SemanticComponentIDs:
    """Exact record IDs for one bundle-backed parser definition snapshot."""

    ontology_modules: Mapping[str, str]
    ontology_bundle: str
    contract: str
    prompts: Mapping[str, str]
    parser: str


def _literal_label(source: bytes, path: str, name: str) -> str:
    # Parse the chosen tree, which may differ from the imported Python package.
    assignments = []
    for node in ast.parse(source, filename=path).body:
        match node:
            case ast.AnnAssign(target=ast.Name(id=target), value=value):
                if target == name:
                    assignments.append(value)
            case ast.Assign(targets=[ast.Name(id=target)], value=value):
                if target == name:
                    assignments.append(value)
    if len(assignments) == 1:
        value = assignments[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    raise ValueError(f"{path} must define {name} as a single literal string")


def _source_set(hashes: Mapping[str, str]) -> str:
    payload = canonical_json(dict(hashes))
    return sha256(b"versioning:source-set:v1\0" + payload).hexdigest()


def _git_producer(
    root: Path, sources: Mapping[str, bytes]
) -> dict[str, object] | None:
    """Attach a commit only when it contains the exact captured parser sources."""

    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL
        )

    try:
        if Path(os.fsdecode(git("rev-parse", "--show-toplevel").rstrip(b"\n"))) != root:
            return None
        commit = git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
        if all(
            git("cat-file", "blob", f"{commit}:{path}") == data
            for path, data in sources.items()
        ):
            return {"git_commit": commit}
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


def _read_definitions(
    root: Path,
) -> tuple[
    dict[str, ComponentRecord],
    ComponentRecord,
    dict[str, tuple[str, str]],
]:
    ontology_dir = root / "parser/ontology"
    manifest_path = ontology_dir / "bundles" / DEFAULT_BUNDLE_MANIFEST.name
    manifest_bytes = manifest_path.read_bytes()
    module_bytes = {
        path.name: path.read_bytes() for path in ontology_dir.glob("*.yaml")
    }

    # Validate and hash the same captured bytes, even if the checkout is edited
    # while the registration is running.
    with TemporaryDirectory() as tmp:
        snapshot = Path(tmp)
        for name, data in module_bytes.items():
            (snapshot / name).write_bytes(data)
        manifest_file = snapshot / "bundles" / manifest_path.name
        manifest_file.parent.mkdir()
        manifest_file.write_bytes(manifest_bytes)
        _, manifest = load_manifest_catalog(manifest_file, ontology_directory=snapshot)

        prompt_configs = {}
        for name, filename in _PROMPT_FILES.items():
            data = (root / "configs/parser_config" / filename).read_bytes()
            staged = snapshot / filename
            staged.write_bytes(data)
            label, _ = _load_prompt_config(staged, name)
            prompt_configs[name] = (label, sha256(data).hexdigest())

    modules = {
        name: ComponentRecord(
            kind="ontology-module",
            name=name,
            label=label,
            label_role="release",
            content_digest=sha256(module_bytes[f"{name}.yaml"]).hexdigest(),
            digest_scope="raw ontology YAML bytes",
        )
        for name, label in sorted(manifest.modules.items())
    }
    bundle = ComponentRecord(
        kind="ontology-bundle",
        name=manifest.bundle_id,
        label=manifest.bundle_version,
        label_role="release",
        content_digest=sha256(manifest_bytes).hexdigest(),
        digest_scope="raw ontology bundle YAML bytes",
        dependencies=tuple(
            Dependency("ontology-module", module.record_id)
            for module in modules.values()
        ),
    )
    return modules, bundle, prompt_configs


def _prompt_records(
    modules: Mapping[str, ComponentRecord],
    contract: ComponentRecord,
    hashes: Mapping[str, str],
    configs: Mapping[str, tuple[str, str]],
    student_label: str,
) -> dict[str, ComponentRecord]:
    contract_ref = Dependency("semantic-contract", contract.record_id)
    ontology_refs = {
        name: Dependency("ontology-module", module.record_id)
        for name, module in modules.items()
    }
    if "discourse" not in ontology_refs or len(ontology_refs) < 2:
        raise ValueError("A semantic snapshot requires discourse and general modules")

    general_refs = tuple(
        ref for key, ref in ontology_refs.items() if key != "discourse"
    )
    prompts = {}
    for name, selected in (
        ("router", ()),
        ("general", general_refs),
        ("discourse", (ontology_refs["discourse"],)),
    ):
        label, yaml_hash = configs[name]
        sources = {
            _TEACHER: hashes[_TEACHER],
            f"configs/parser_config/{_PROMPT_FILES[name]}": yaml_hash,
        }
        prompts[name] = ComponentRecord(
            kind="teacher-prompt",
            name=name,
            label=label,
            label_role="family",
            content_digest=_source_set(sources),
            digest_scope="prompt definition source set v1",
            dependencies=(contract_ref, *selected),
            details={"source_sha256": sources},
        )
    routed_sources = {
        _TEACHER: hashes[_TEACHER],
        **{
            f"configs/parser_config/{_PROMPT_FILES[name]}": configs[name][1]
            for name in ("general", "discourse")
        },
    }
    prompts["routed"] = ComponentRecord(
        kind="teacher-prompt",
        name="routed",
        content_digest=_source_set(routed_sources),
        digest_scope="prompt definition source set v1",
        dependencies=(
            contract_ref,
            Dependency("general-prompt", prompts["general"].record_id),
            Dependency("discourse-prompt", prompts["discourse"].record_id),
        ),
        details={"source_sha256": routed_sources},
    )
    student_sources = {_STUDENT: hashes[_STUDENT]}
    prompts["student"] = ComponentRecord(
        kind="student-prompt",
        name="student",
        label=student_label,
        label_role="family",
        content_digest=_source_set(student_sources),
        digest_scope="prompt definition source set v1",
        dependencies=(contract_ref, *ontology_refs.values()),
        details={"source_sha256": student_sources},
    )
    return prompts


def register_semantic_components(
    registry: Registry,
    *,
    project_root: str | Path = ".",
    include_git_provenance: bool = False,
) -> SemanticComponentIDs:
    """Register the default bundle, contract, prompts, and parser from a checkout."""
    root = Path(project_root).resolve()
    modules, bundle, prompt_configs = _read_definitions(root)
    sources = {
        path: (root / path).read_bytes()
        for path in (*_PARSER_SOURCES, _SCHEMA, _TEACHER, _STUDENT)
    }
    hashes = {path: sha256(data).hexdigest() for path, data in sources.items()}
    contract = ComponentRecord(
        kind="parser-contract",
        name="semantic",
        label=_literal_label(sources[_SCHEMA], _SCHEMA, "SEMANTIC_CONTRACT_VERSION"),
        label_role="locator",
        content_digest=hashes[_SCHEMA],
        digest_scope="raw semantic schema Python source bytes",
        details={"source_path": _SCHEMA},
    )
    prompts = _prompt_records(
        modules,
        contract,
        hashes,
        prompt_configs,
        _literal_label(sources[_STUDENT], _STUDENT, "STUDENT_PROMPT"),
    )
    parser_sources = {path: hashes[path] for path in _PARSER_SOURCES}
    parser = ComponentRecord(
        kind="parser-implementation",
        name="semantic",
        label=_literal_label(sources[_PARSER], _PARSER, "PARSER_VERSION"),
        label_role="locator",
        content_digest=_source_set(parser_sources),
        digest_scope="parser implementation source set v1",
        dependencies=(
            Dependency("semantic-contract", contract.record_id),
            Dependency("ontology-bundle", bundle.record_id),
            *(
                Dependency(f"{name}-prompt", record.record_id)
                for name, record in prompts.items()
            ),
        ),
        producer=(
            _git_producer(root, {path: sources[path] for path in _PARSER_SOURCES})
            if include_git_provenance
            else None
        ),
        details={"source_sha256": parser_sources},
    )
    for record in (*modules.values(), bundle, contract, *prompts.values(), parser):
        registry.register(record)
    return SemanticComponentIDs(
        ontology_modules=MappingProxyType(
            {name: record.record_id for name, record in modules.items()}
        ),
        ontology_bundle=bundle.record_id,
        contract=contract.record_id,
        prompts=MappingProxyType(
            {name: record.record_id for name, record in prompts.items()}
        ),
        parser=parser.record_id,
    )
