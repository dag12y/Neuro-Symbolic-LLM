"""Observe local symbolic exports and register their completed artifacts."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Never, cast

from .nodes import Dependency, Node, reachable_nodes, validate_id
from .symbolic import build_symbolic_definition, validate_symbolic_definition

if TYPE_CHECKING:
    from .registry import Registry

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BUNDLE = Path("parser/ontology/bundles/semantic-extraction-v1.yaml")
_ADAPTER_BUNDLE = "ontology/miner-bundle"
_PARSER_BACKEND = "configuration/parser-backend"
_PARSER_SUPPORT = (
    "parser/semantic/catalog_document_export.py",
    "parser/semantic/document_metadata.py",
)


def _projection_value(node: Node) -> object:
    return cast(dict[str, object], node.to_dict()["payload"])["value"]


@dataclass(frozen=True)
class RunResult:
    root: Node
    accepted: int
    rejected: int
    skipped_rules: int


@dataclass(frozen=True)
class _Produced:
    paths: Mapping[str, Path | None]
    accepted: int
    rejected: int
    skipped_rules: int = 0
    bundle_bytes: bytes | None = None


def _file_identity(
    path: Path, copy_to: Path | None = None, *, follow_links: bool = True
) -> dict[str, object]:
    """Stream bytes from a regular file, optionally making the producer's copy."""
    digest = sha256()
    length = 0

    def signature(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    with ExitStack() as stack:
        source = stack.enter_context(
            os.fdopen(
                os.open(
                    path,
                    os.O_RDONLY
                    | os.O_NONBLOCK
                    | (0 if follow_links else os.O_NOFOLLOW),
                ),
                "rb",
            )
        )
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        target = (
            stack.enter_context(copy_to.open("xb")) if copy_to is not None else None
        )
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            length += len(chunk)
            if target is not None:
                target.write(chunk)
        if length != before.st_size or signature(before) != signature(path.stat()):
            raise ValueError(f"File changed while reading: {path}")
    return {"sha256": digest.hexdigest(), "byte_length": length}


def _jsonl_count(path: Path, *, skipped: bool = False) -> int:
    def invalid_constant(token: str) -> Never:
        raise ValueError(f"Unsupported JSON constant: {token}")

    count = 0
    with path.open("rb") as file:
        for number, line in enumerate(file, 1):
            try:
                value = json.loads(
                    line.decode("utf-8"), parse_constant=invalid_constant
                )
                if not line.endswith(b"\n") or not isinstance(value, dict):
                    raise ValueError("expected a complete JSON object line")
                if skipped:
                    rules = value.get("skipped_rules")
                    if (
                        not isinstance(rules, list)
                        or not rules
                        or not all(isinstance(rule, str) for rule in rules)
                    ):
                        raise ValueError("expected a nonempty list of skipped rules")
                    count += len(rules)
                else:
                    count += 1
            except ValueError as error:
                raise ValueError(
                    f"Invalid output artifact {path.name}:{number}"
                ) from error
    return count


def _observe_outputs(
    paths: dict[str, Path], produced: _Produced
) -> dict[str, dict[str, object]]:
    returned = {role: path for role, path in produced.paths.items() if path is not None}
    if returned != paths:
        raise ValueError("Exporter returned a different artifact set")
    directory = paths["accepted"].parent
    if set(directory.iterdir()) != set(paths.values()) or any(
        path.is_symlink() or not path.is_file() for path in paths.values()
    ):
        raise ValueError("Missing or unexpected materialized export artifacts")
    counts = {
        "accepted": produced.accepted,
        "rejected": produced.rejected,
        "metadata": produced.accepted,
        "student-pairs": produced.accepted,
        "skipped-rules": produced.skipped_rules,
    }
    if produced.skipped_rules and "skipped-rules" not in paths:
        raise ValueError(
            "Exporter reported skipped rules without a skipped-rule artifact"
        )
    identities = {}
    for role, path in paths.items():
        identities[role] = _file_identity(path)
        if role == "ontology-bundle":
            expected = produced.bundle_bytes
            if expected is None or identities[role] != {
                "sha256": sha256(expected).hexdigest(),
                "byte_length": len(expected),
            }:
                raise ValueError(
                    "Materialized ontology bundle differs from effective bundle"
                )
        elif _jsonl_count(path, skipped=role == "skipped-rules") != counts[role]:
            raise ValueError(f"Exporter count does not match {role} artifact")
    return identities


def _record_run(
    definition: tuple[Node, ...],
    recipe: str,
    source: Node,
    support: tuple[Node, ...],
    effective: Node,
    options: dict[str, object],
    produced: _Produced,
    identities: dict[str, dict[str, object]],
    registry: Registry,
    runtime: dict[str, object] | None = None,
) -> RunResult:
    prefix = f"symbolic/m3/{recipe}"

    def composite(stage: str, purpose: str, edges: tuple[Dependency, ...]) -> Node:
        name = f"{prefix}/{stage}"
        return Node(
            kind="composite",
            name=name,
            payload={"recipe": {"name": name, "version": 1}, "purpose": purpose},
            dependencies=edges,
        )

    def metadata(stage: str, value: dict[str, object]) -> Node:
        return Node(
            kind="semantic_content",
            name=f"{prefix}/{stage}",
            payload={"projection": f"{recipe}-{stage}", "version": 1, "value": value},
        )

    options_node = metadata("options", options)
    invocation = composite(
        "inputs",
        "component",
        (
            Dependency("uses-component", "definition", definition[-1].id),
            Dependency("uses-source", "input", source.id),
            Dependency("uses-semantic", "effective-declaration", effective.id),
            Dependency("uses-semantic", "options", options_node.id),
            *(
                Dependency("uses-source", f"producer/{node.name}", node.id)
                for node in support
            ),
        ),
    )
    artifacts = {
        role: Node(
            kind="artifact",
            name=f"{prefix}/output/{role}",
            payload=identity,
            dependencies=(Dependency("produced-by", "inputs", invocation.id),),
        )
        for role, identity in identities.items()
    }
    outcome = metadata(
        "outcome",
        {
            "status": "completed_with_rejections" if produced.rejected else "completed",
            "accepted": produced.accepted,
            "rejected": produced.rejected,
            "skipped_rules": produced.skipped_rules,
        },
    )
    run = composite(
        "run",
        "run",
        (
            Dependency("uses-component", "inputs", invocation.id),
            Dependency("uses-semantic", "outcome", outcome.id),
            *(
                Dependency("uses-artifact", role, node.id)
                for role, node in artifacts.items()
            ),
        ),
    )
    batch = (
        *definition,
        source,
        *support,
        options_node,
        invocation,
        *artifacts.values(),
        outcome,
        run,
    )
    validate_symbolic_run(batch, run.id)
    if runtime is not None:
        from .parser_observation import observed_parser, validate_parser

        extra = observed_parser(run, runtime)
        batch = (*batch, *extra)
        run = extra[-1]
        validate_parser(run, {n.id: n for n in batch})
    registry.register_batch(batch)
    return RunResult(run, produced.accepted, produced.rejected, produced.skipped_rules)


def _execute(
    input_path: Path,
    output_dir: Path,
    registry: Registry,
    *,
    recipe: str,
    source_name: str,
    filenames: dict[str, str],
    options: dict[str, object],
    effective_name: str,
    producer: Callable[[Path, dict[str, Path], Mapping[str, Node]], _Produced],
    expected_definition: str | None,
    sources: tuple[str, ...] = (),
    runtime: dict[str, object] | None = None,
) -> RunResult:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {output_dir}")
    database = registry.path.resolve()
    sqlite_paths = (
        database,
        *(Path(str(database) + suffix) for suffix in ("-journal", "-wal", "-shm")),
    )
    if any(
        path.is_relative_to(output_dir.resolve())
        or (path.exists() and path.samefile(input_path))
        for path in sqlite_paths
    ):
        raise ValueError("Registry must be separate from the input and export outputs")
    definition = build_symbolic_definition(_PROJECT_ROOT)
    if expected_definition is not None and definition[-1].id != expected_definition:
        raise ValueError("Current symbolic definition differs from expected definition")
    by_name = {node.name: node for node in definition}
    support = tuple(
        Node(kind="source", name=name, payload=_file_identity(_PROJECT_ROOT / name))
        for name in sources
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".symbolic-export-", dir=output_dir.parent
    ) as directory:
        scratch = Path(directory)
        private_input = scratch / source_name
        identity = _file_identity(input_path, copy_to=private_input)
        staged = scratch / "outputs"
        staged.mkdir()
        paths = {role: staged / filename for role, filename in filenames.items()}
        try:
            produced = producer(private_input, paths, by_name)
        except Exception as error:
            if runtime is None:
                raise
            from .parser_observation import failed_parser

            if (
                _file_identity(private_input) != identity
                or build_symbolic_definition(_PROJECT_ROOT) != definition
                or any(
                    _file_identity(_PROJECT_ROOT / node.name) != node.payload
                    for node in support
                )
            ):
                raise ValueError(
                    "Parser inputs changed during failed execution"
                ) from error
            failed_parser(
                registry,
                definition,
                support,
                source_name,
                identity,
                options,
                runtime,
                error,
            )
            raise
        identities = _observe_outputs(paths, produced)
        if build_symbolic_definition(_PROJECT_ROOT)[-1].id != definition[-1].id or any(
            _file_identity(_PROJECT_ROOT / node.name) != node.payload
            for node in support
        ):
            raise ValueError(
                "Symbolic definition or producer source changed during export"
            )
        if any(
            _file_identity(path) != identity for path in (input_path, private_input)
        ):
            raise ValueError("Input changed during export")

        # Hard-link complete files into a new directory on the same filesystem.
        # Neither directory creation nor linking can replace an existing path.
        output_dir.mkdir(mode=0o700)
        directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for role, path in paths.items():
                os.link(path, filenames[role], dst_dir_fd=directory_fd)
            for role, filename in filenames.items():
                if (
                    _file_identity(output_dir / filename, follow_links=False)
                    != identities[role]
                ):
                    raise ValueError(f"Published export artifact changed: {role}")
            if not os.path.samestat(os.fstat(directory_fd), output_dir.lstat()):
                raise ValueError("Published directory changed during export")
        finally:
            os.close(directory_fd)
        source = Node(
            kind="source",
            name=f"symbolic/m3/{recipe}/input/{source_name}",
            payload=identity,
        )
        return _record_run(
            definition,
            recipe,
            source,
            support,
            by_name[effective_name],
            options,
            produced,
            identities,
            registry,
            runtime,
        )


def record_symbolic_export_run(
    input_path: Path,
    output_dir: Path,
    registry: Registry,
    *,
    bundle_output: bool = False,
    skip_unsupported_rules: bool = False,
    expected_definition: str | None = None,
) -> RunResult:
    """Execute the real parser-result to miner-document exporter and record its run."""
    from hybrid_miner.ontology.bundle import OntologyBundle

    from symbolic_pipeline.document_export import export_semantic_documents
    from symbolic_pipeline.miner_bundle import load_catalog_bundle

    filenames = {"accepted": "documents.jsonl", "rejected": "rejected.jsonl"}
    if bundle_output:
        filenames["ontology-bundle"] = "ontology-bundle.json"
    if skip_unsupported_rules:
        filenames["skipped-rules"] = "skipped-rules.jsonl"

    def produce(
        input_copy: Path, paths: dict[str, Path], captured: Mapping[str, Node]
    ) -> _Produced:
        _, bundle = load_catalog_bundle(
            _PROJECT_ROOT / _DEFAULT_BUNDLE,
            ontology_directory=_PROJECT_ROOT / "parser/ontology",
        )
        expected = OntologyBundle.model_validate(
            _projection_value(captured[_ADAPTER_BUNDLE])
        )
        if bundle.canonical_json() != expected.canonical_json():
            raise ValueError("Loaded ontology bundle differs from captured definition")
        result = export_semantic_documents(
            input_copy,
            paths["accepted"],
            paths["rejected"],
            bundle=bundle,
            bundle_path=paths.get("ontology-bundle"),
            skipped_path=paths.get("skipped-rules"),
        )
        return _Produced(
            {
                "accepted": result.output_path,
                "rejected": result.rejected_path,
                "ontology-bundle": result.bundle_path,
                "skipped-rules": result.skipped_path,
            },
            result.accepted,
            result.rejected,
            result.skipped_rules,
            (bundle.canonical_json() + os.linesep).encode("utf-8"),
        )

    return _execute(
        input_path,
        output_dir,
        registry,
        recipe="adapter-export",
        source_name="parser-results.jsonl",
        filenames=filenames,
        options={
            "bundle_output": bundle_output,
            "skip_unsupported_rules": skip_unsupported_rules,
        },
        effective_name=_ADAPTER_BUNDLE,
        producer=produce,
        expected_definition=expected_definition,
    )


def record_parser_export_run(
    input_path: Path,
    output_dir: Path,
    registry: Registry,
    *,
    observe_runtime: bool = False,
    corpus_id: str | None = None,
    include_metta: bool = False,
    asserted_only: bool = False,
    student_pairs: bool = False,
    require_structured_output: bool = False,
    expected_definition: str | None = None,
) -> RunResult:
    """Execute a fresh parser export using the tracked active reference profile."""
    from parser.semantic.catalog_document_export import export_catalog_documents
    from parser.semantic.model_config import build_reference_semantic_parser
    from parser.semantic.semantic_parser import ModelGenerationError

    runtime: dict[str, object] | None = {} if observe_runtime else None
    filenames = {
        "accepted": "parser-results.jsonl",
        "rejected": "rejected.jsonl",
        "metadata": "metadata.jsonl",
    }
    if student_pairs:
        filenames["student-pairs"] = "student-pairs.jsonl"
    source_format = "jsonl" if input_path.suffix.lower() == ".jsonl" else "txt"
    effective_corpus = corpus_id or input_path.stem

    def produce(
        input_copy: Path, paths: dict[str, Path], captured: Mapping[str, Node]
    ) -> _Produced:
        from parser.grammar import atomese

        from .semantic import backend_projection, load_catalog_inputs

        grammar = cast(dict[str, Any], _projection_value(captured["grammar/atomese"]))
        vocabulary = cast(
            dict[str, Any], _projection_value(captured["ontology/default-vocabulary"])
        )
        fixed = {**grammar["fixed"], **vocabulary["arities"]}
        variable = {name: tuple(bounds) for name, bounds in grammar["variable"].items()}
        if (
            fixed != atomese.ARITY
            or variable != atomese.VARIABLE_ARITY
            or set(fixed) | set(variable) != atomese.PREDICATES
        ):
            raise ValueError("Loaded Atomese grammar differs from captured definition")

        config = _PROJECT_ROOT / "configs/parser_config/model_backend.yaml"
        config_copy = input_copy.parent / "model-backend.yaml"
        _file_identity(config, copy_to=config_copy)
        if backend_projection(config_copy) != _projection_value(
            captured[_PARSER_BACKEND]
        ):
            raise ValueError("Active parser profile differs from captured definition")
        catalogs = load_catalog_inputs(
            (_PROJECT_ROOT / "parser/ontology").glob("*.yaml"),
            _PROJECT_ROOT / _DEFAULT_BUNDLE,
            None,
        )
        selection = {
            "modules": list(catalogs.selection),
            "versions": list(catalogs.selected.versions),
        }
        if selection != _projection_value(captured["ontology/selected-catalog"]) or any(
            catalogs.modules[name]
            != _projection_value(captured[f"ontology/module/{name}"])
            for name in catalogs.selection
        ):
            raise ValueError("Loaded parser catalog differs from captured definition")
        parser = build_reference_semantic_parser(
            config_copy,
            catalog=catalogs.selected,
            require_structured_output=require_structured_output,
        )
        if runtime is not None:
            backend = getattr(parser, "_backend", None)
            model = getattr(backend, "_model", None)
            runtime.update(
                model=parser.model_name,
                backend=type(backend).__name__,
                revision_reported=getattr(
                    getattr(model, "config", None), "_commit_hash", None
                ),
            )
        try:
            result = export_catalog_documents(
                parser,
                input_path=input_copy,
                output_path=paths["accepted"],
                rejected_path=paths["rejected"],
                metadata_output=paths["metadata"],
                student_pairs_path=paths.get("student-pairs"),
                corpus_id=effective_corpus,
                include_metta=include_metta,
                asserted_only=asserted_only,
            )
        except ModelGenerationError as error:
            raise ValueError("Reference parser generation failed") from error
        return _Produced(
            {
                "accepted": result.output_path,
                "rejected": result.rejected_path,
                "metadata": result.metadata_path,
                "student-pairs": result.student_pairs_path,
            },
            result.accepted,
            result.rejected,
        )

    return _execute(
        input_path,
        output_dir,
        registry,
        recipe="parser-export",
        source_name=f"source.{source_format}",
        filenames=filenames,
        options={
            "corpus_id_sha256": sha256(effective_corpus.encode("utf-8")).hexdigest(),
            "include_metta": include_metta or asserted_only,
            "asserted_only": asserted_only,
            "student_pairs": student_pairs,
            "require_structured_output": require_structured_output,
        },
        effective_name=_PARSER_BACKEND,
        producer=produce,
        expected_definition=expected_definition,
        sources=_PARSER_SUPPORT,
        runtime=runtime,
    )


def validate_symbolic_run(nodes: Iterable[Node], root_id: str) -> tuple[Node, ...]:
    """Check a complete M3 closure, including its shared M2b definition."""
    index = {node.id: node for node in nodes}
    closure = reachable_nodes((root_id,), index.__getitem__)
    named = {node.name: node for node in closure}
    if len(named) != len(closure):
        raise ValueError("Conflicting logical names in run closure")
    root = closure[-1]
    prefix, _, stage = root.name.rpartition("/")
    recipe = prefix.rsplit("/", 1)[-1]
    parser = recipe == "parser-export"
    if stage != "run" or prefix not in {
        "symbolic/m3/parser-export",
        "symbolic/m3/adapter-export",
    }:
        raise ValueError("Unknown M3 run recipe")

    def node(name: str) -> Node:
        if name not in named:
            raise ValueError(f"Missing required run node: {name}")
        return named[name]

    def composite(item: Node, purpose: str) -> None:
        if item.kind != "composite" or item.payload != {
            "purpose": purpose,
            "recipe": {"name": item.name, "version": 1},
        }:
            raise ValueError(f"Invalid composite recipe: {item.name}")

    def edges(item: Node, required: Iterable[Dependency]) -> None:
        if set(item.dependencies) != set(required):
            raise ValueError(f"Invalid dependency slots or targets: {item.name}")

    def projection(label: str) -> Mapping[str, object]:
        item = node(f"{prefix}/{label}")
        value = item.payload.get("value")
        if (
            item.kind != "semantic_content"
            or item.dependencies
            or item.payload.get("projection") != f"{recipe}-{label}"
            or item.payload.get("version") != 1
            or not isinstance(value, Mapping)
        ):
            raise ValueError(f"Invalid {label} projection")
        return value

    composite(root, "run")
    invocation = node(f"{prefix}/inputs")
    composite(invocation, "component")
    definition = node("symbolic/definition")
    validate_symbolic_definition(closure, definition.id)
    if definition.payload["recipe"] != {
        "name": "symbolic/m2b/definition",
        "version": 1,
    }:
        raise ValueError("Run requires an M2b definition")

    options = projection("options")
    flags = (
        {"include_metta", "asserted_only", "student_pairs", "require_structured_output"}
        if parser
        else {"bundle_output", "skip_unsupported_rules"}
    )
    if set(options) != flags | ({"corpus_id_sha256"} if parser else set()) or any(
        not isinstance(options[k], bool) for k in flags
    ):
        raise ValueError("Invalid run options")
    if parser:
        validate_id(options["corpus_id_sha256"])
        if options["asserted_only"] and not options["include_metta"]:
            raise ValueError("asserted_only requires effective MeTTa output")
    sources = ("source.txt", "source.jsonl") if parser else ("parser-results.jsonl",)
    inputs = [
        named[f"{prefix}/input/{name}"]
        for name in sources
        if f"{prefix}/input/{name}" in named
    ]
    if len(inputs) != 1:
        raise ValueError("Expected one named input")
    edges(
        invocation,
        (
            Dependency("uses-source", "input", inputs[0].id),
            Dependency("uses-component", "definition", definition.id),
            Dependency("uses-semantic", "options", node(f"{prefix}/options").id),
            Dependency(
                "uses-semantic",
                "effective-declaration",
                node(_PARSER_BACKEND if parser else _ADAPTER_BUNDLE).id,
            ),
            *(
                Dependency("uses-source", f"producer/{name}", node(name).id)
                for name in (_PARSER_SUPPORT if parser else ())
            ),
        ),
    )

    outcome = projection("outcome")
    counts = ("accepted", "rejected", "skipped_rules")
    if set(outcome) != {"status", *counts}:
        raise ValueError("Invalid outcome fields")
    totals: dict[str, int] = {}
    for key in counts:
        count = outcome[key]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"Invalid outcome count: {key}")
        totals[key] = count
    if outcome["status"] != (
        "completed_with_rejections" if totals["rejected"] else "completed"
    ):
        raise ValueError("Outcome status disagrees with rejected count")
    roles = {"accepted": totals["accepted"], "rejected": totals["rejected"]}
    if parser:
        roles["metadata"] = totals["accepted"]
        if options["student_pairs"]:
            roles["student-pairs"] = totals["accepted"]
    elif options["bundle_output"]:
        roles["ontology-bundle"] = 1
    if options.get("skip_unsupported_rules"):
        roles["skipped-rules"] = totals["skipped_rules"]
    if totals["skipped_rules"] and (
        "skipped-rules" not in roles or not totals["accepted"]
    ):
        raise ValueError("Inconsistent skipped-rule count")
    required = [
        Dependency("uses-component", "inputs", invocation.id),
        Dependency("uses-semantic", "outcome", node(f"{prefix}/outcome").id),
    ]
    for role, count in roles.items():
        artifact = node(f"{prefix}/output/{role}")
        edges(artifact, (Dependency("produced-by", "inputs", invocation.id),))
        length = cast(int, artifact.payload["byte_length"])
        if not length and artifact.payload["sha256"] != sha256(b"").hexdigest():
            raise ValueError(f"Invalid empty artifact digest: {role}")
        if bool(length) != bool(count) or count > length:
            raise ValueError(f"Artifact length disagrees with outcome: {role}")
        required.append(Dependency("uses-artifact", role, artifact.id))
    edges(root, required)
    return closure


def verify_run_artifacts(
    nodes: Iterable[Node], root_id: str, paths: Mapping[str, Path]
) -> None:
    """Validate the recipe and compare explicitly supplied regular files with its outputs."""
    closure = validate_symbolic_run(nodes, root_id)
    index = {node.id: node for node in closure}
    artifacts = {
        edge.slot: index[edge.target_id]
        for edge in closure[-1].dependencies
        if edge.role == "uses-artifact"
    }
    if set(paths) != set(artifacts):
        raise ValueError(
            f"Artifact roles differ: missing {sorted(artifacts.keys() - paths.keys())}, unexpected {sorted(paths.keys() - artifacts.keys())}"
        )
    for role, artifact in artifacts.items():
        try:
            actual = _file_identity(paths[role], follow_links=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot verify artifact {role}: {error}") from error
        if actual != artifact.payload:
            raise ValueError(f"Artifact digest or length mismatch: {role}")
