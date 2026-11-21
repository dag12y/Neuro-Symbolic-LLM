# Merkle DAG foundation

A node identifies a source file, a declared external Git release, or a
composition of other nodes. A definition root is a composite with purpose
`definition`. Dependencies point from a consumer to prerequisites. Two
consumers can share the same prerequisite ID.

## Identity

The hashed object contains exactly `schema_version`, `canonicalization_version`,
`kind`, `name`, `payload`, and `dependencies`. Both version fields are `1`.
The serialized node adds the computed `id`.

| Kind | Payload |
| --- | --- |
| `source` | `sha256` over exact file bytes and nonnegative integer `byte_length`. The node name is a repository-relative logical path. |
| `external_release` | `repository` is a normalized HTTPS Git locator with a lowercase hostname and case-preserving path. The node name is the lowercase, hyphen-separated package name. `selector` contains exactly `kind` (`tag` or `commit`) and `value` (a tag or a full lowercase Git object ID). Both selector kinds are declarations. |
| `composite` | `recipe` contains a logical `name` and positive integer `version`. `purpose` is `component`, `definition`, or `run`. A composite has at least one dependency. |
| `semantic_content` | `projection` names the interpretation, `version` is a positive integer, and `value` contains the projected JSON object. This M2b extension uses the existing hash domain and manifest envelope. Previously published nodes retain their IDs. |

An edge contains exactly `role`, `slot`, and `target_id`. The roles
`uses-source`, `uses-release`, and `uses-component` require targets of kinds
`source`, `external_release`, and `composite`, respectively. M2b adds
`uses-semantic` for `semantic_content` targets. Slots are logical
names, unique within a role. Edges sort by `(role, slot)`. Source and release
nodes are leaves. A recipe can assign meaning to slots, including explicit
ordering with indexed names.

The ID is lowercase hex SHA-256 of the bytes `nsllm:merkle-node:v1`, one NUL
byte, and the canonical identity object. JSON keys sort by Unicode code point.
Arrays preserve order. There is no additional whitespace. Strings use compact
JSON escaping with ASCII output encoded as UTF-8, lowercase hexadecimal
Unicode escapes, and surrogate pairs for supplementary characters. There is no
Unicode normalization. Numbers are signed 64-bit integers written in decimal.
Floating-point values, duplicate keys, unsupported JSON values, and lone
surrogates are rejected. `hash_file(path)` separately streams raw SHA-256.

Golden canonical bytes, with no trailing newline:

```json
{"a":[true,null,-9223372036854775808,9223372036854775807],"z":0,"\ue000":"\u00e9\n","\ud83d\ude00":"\u007f"}
{"canonicalization_version":1,"dependencies":[],"kind":"source","name":"parser/example.py","payload":{"byte_length":3,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},"schema_version":1}
```

The domain-separated digests are, respectively:

```text
64638a477b01c0f8bb61498b0e358bad55a78b81a90f0bd1892cc50aac9226e5
e93b978a7df4977e71d33d713d2ed159f457bb8c88e8ed0cf41e8a0e9600f6b0
```

## Register and inspect a fixture

Run this from the repository root. The fixture uses hardcoded declarations and
does not read project configuration or contact the network.

```bash
demo_dir=$(mktemp -d)
root_id=$(python - "$demo_dir/graph.sqlite3" <<'PY'
from hashlib import sha256
from sys import argv

from versioning.nodes import DraftDependency, DraftNode, build_nodes
from versioning.registry import Registry

def stage(name, dependencies, purpose="component"):
    return DraftNode(
        kind="composite",
        name=f"symbolic/{name}",
        payload={"recipe": {"name": "fixture/symbolic", "version": 1}, "purpose": purpose},
        dependencies=tuple(dependencies),
    )

def fixture(tag):
    return build_nodes({
        "source": DraftNode(
            kind="source",
            name="parser/example.py",
            payload={"sha256": sha256(b"abc").hexdigest(), "byte_length": 3},
        ),
        "release": DraftNode(
            kind="external_release",
            name="hybrid-miner",
            payload={
                "repository": "https://github.com/iCog-Labs-Dev/hybrid-miner.git",
                "selector": {"kind": "tag", "value": tag},
            },
        ),
        "parser": stage("parser", [DraftDependency("uses-source", "implementation", "source")]),
        "adapter": stage("adapter", [DraftDependency("uses-release", "miner", "release")]),
        "mining": stage("mining", [DraftDependency("uses-release", "miner", "release")]),
        "root": stage(
            "pipeline",
            [DraftDependency("uses-component", name, name) for name in ("parser", "adapter", "mining")],
            "definition",
        ),
    })

registry = Registry(argv[1])
old, new = fixture("v0.1.0"), fixture("v0.1.1")
for graph in (old, new):
    registry.register_batch(graph.values())
assert {name for name in old if old[name].id != new[name].id} == {
    "release", "adapter", "mining", "root",
}
assert registry.get(old["root"].id) == old["root"]
print(new["root"].id)
PY
)

python -m versioning inspect --registry "$demo_dir/graph.sqlite3" "$root_id"
python -m versioning export-manifest --registry "$demo_dir/graph.sqlite3" \
    "$root_id" > "$demo_dir/manifest.json"
python -m versioning verify-manifest "$demo_dir/manifest.json" --expected-root "$root_id"
```

`build_nodes` resolves named draft dependencies and hashes bottom-up. A caller
with existing node IDs can construct nodes directly with `Dependency` and
register a batch that reuses those stored targets. SQLite publishes a complete
batch in one write transaction. Registering identical nodes again leaves their
IDs and stored bytes unchanged. Old roots remain addressable by full ID.

## Manifest verification

A version 2 manifest contains exactly `manifest_version`, `root_id`, and the
reachable `nodes` sorted by ID. Verification works without the registry. It
validates the graph and recomputes identities. When supplied independently,
`--expected-root` also checks that the manifest matches the selected root.
Verification does not read source file bytes or resolve external references.
Use `hash_file` separately to compare supplied files with their recorded
digests. A verified graph can contain an unresolved external declaration.

## Snapshot the symbolic definition

`snapshot-symbolic` reads `[project.optional-dependencies].miner` from the
selected project's `pyproject.toml` with `tomllib`. It requires exactly one
`hybrid-miner` Git direct requirement pointing to
`https://github.com/iCog-Labs-Dev/hybrid-miner.git`. Case variants of the GitHub
locator, an omitted `.git` suffix, and package-name variants normalize to the
same identity. It accepts bare
version tags such as `v0.1.1`, fully qualified `refs/tags/<tag>` selectors,
and full 40- or 64-hex-digit commit selectors. The project convention treats
bare version-shaped selectors as tags. Branch names, short commit IDs,
wildcards, markers, and unexpected origins fail capture.

The release node hashes the normalized declaration. Other TOML fields and
installed package metadata do not contribute to it. The snapshot needs neither
a miner installation nor network access. A tag or commit in the manifest is
a declared selector, with source contents and tag-to-commit resolution outside
the scope of verification.

```bash
root_id=$(python -m versioning snapshot-symbolic --project-root . --registry graph.sqlite3)
python -m versioning inspect --registry graph.sqlite3 "$root_id"
python -m versioning export-manifest --registry graph.sqlite3 "$root_id" > manifest.json
python -m versioning verify-manifest manifest.json --expected-root "$root_id"
```

The CLI prints only the root ID after atomic registration. The default recipe
is M2b. Pass `--recipe m2a` to reproduce the historical source-only recipe.
Both recipes can coexist in the same registry, and existing M2a manifests
continue to verify. The generic verifier checks hashes and graph structure.
`versioning.symbolic.validate_symbolic_definition(nodes, root_id)` checks the
symbolic recipe, including exact M2b semantic projection labels and versions,
required dependency roles and targets, shared sources, and selected catalog
membership and versions. It rejects unexpected reachable semantic nodes. It
validates the recipe shape and declarations from the manifest. Checking the
declared values against original source files still requires those files.

### M2b inventory

The root depends on the parser, adapter, and external mining definitions.
The default parser and adapter use `parser/ontology/bundles/semantic-extraction-v1.yaml`.
The parser can select another bundle with `--ontology-bundle` (paths are relative
to `--project-root` unless absolute). The adapter still uses its default bundle,
and Atomese still loads the full default ontology directory.

| Input | Captured identity and consumers |
| --- | --- |
| Ontology modules and selected catalog | The default bundle selects `discourse`, `general`, `lexical`, `narrative`, `program`, `proof`, and `upper`. `upper` declares shared entity, event, proposition, and clause types. Selected modules have individual semantic IDs. The catalog uses ordered membership and versions plus its loader source. |
| Prompts | Router, general, and discourse YAML contribute their loaded version and exact template. General and discourse prompts depend on their generated ontology guides from the selected catalog. The student prompt includes generated instructions from its builder. Python builder sources capture the formatting and fixed routing/combined instructions. |
| Atomese grammar | Effective fixed and variable arities from `atomese_grammar.yaml`, the full ontology directory's predicate names and arities, and `parser/grammar/atomese.py`. Grammar descriptions and roles are ignored by the runtime arity loader. |
| Backend declaration | The active reference backend profile from `model_backend.yaml` and `model_config.py`. This includes its provider, model locator or environment variable name, effective generation options, and credential/host variable names used by that provider. Inactive profiles and actual environment values are excluded. |
| Parser implementation | Raw Python bytes for `semantic_parser.py`, `schema.py`, `backends.py`, `metta_renderer.py`, `normalization.py`, `output_cleanup.py`, and `catalog_student.py`, with catalog, grammar, and prompt builders linked from their projections. |
| Adapter and mining | The adapter uses raw bytes of `miner_adapter.py`, `miner_bundle.py`, `document_export.py`, and the shared parser schema, plus the projected input to the miner's bundle validator. That projection contains the default bundle declaration's ID and version, catalog versions and types, and ordered predicate signatures. The adapter and external mining stage share the declared release node. |

The ontology and manifest projections validate the captured text with the
runtime catalog loaders. Keys use canonical JSON ordering. The ontology
loader's normalization of type names, argument names, type unions, and
extraction instructions is retained. Type declarations are a set. Argument
positions, types in each union, confidence bounds, and catalog version order
are significant. Catalog order comes from sorted module filenames, so
reordering keys in a bundle manifest leaves its identity stable. Confidence
bounds and backend temperatures use exact big-endian IEEE-754 binary64 hex
strings within the JSON projection, including signed zero.

YAML comments and presentation changes that preserve loaded values leave the
semantic IDs unchanged. A `#` inside a block scalar is part of the text.
Snapshot capture rejects duplicate or non-string mapping keys, anchors,
aliases, merge keys, and tags outside the supported JSON-like scalar and
collection types. That strict subset keeps ambiguous or unsupported inputs
out of the registry. Raw Python byte changes, including comments, change the
corresponding source node. Use `hash_file` for exact YAML byte comparisons.
Raw YAML hashes are separate from these semantic graphs.

For example, a general ontology instruction change affects the module,
selected catalog, general guide and prompt, student instructions, parser,
and root. The miner bundle and Atomese vocabulary stay stable if their
inputs are unchanged. An argument-type union reorder also affects the bundle
signature. An unselected module's relation arity still affects Atomese. A
new miner release affects the adapter, mining, and root, with a stable parser
ID. The shared parser schema affects both parser and adapter.
`versioning.semantic.compare_parser_definitions(before, after)` reports named
additions, removals, and modifications in parser closures or verified manifests.

Capture uses the projection rules shipped with the tracker and reads sources
from a working directory. Keep that directory stable during capture. The
tracker hashes changes to loader source and does not execute arbitrary
loader code from the captured directory. This is a definition snapshot.
It excludes CLI overrides beyond the bundle selector, environment values,
actual model or miner contents, and execution inputs and outputs. A
definition change does not establish a compatibility change.

The backend projection follows `build_reference_semantic_parser()` and
`create_backend()`. It captures the active profile's provider, model or
model environment-variable name, provider-specific environment-variable
names and token limit, Ollama options, Transformers' `add_special_tokens`
setting, and the explicit `role: student` check. Reference mode can select
the student-named profile, in which case it follows the same reference
factory. It reads no environment values, credentials, or local checkpoint
metadata. The separate `build_distilled_semantic_parser()` entry point
defaults to `semantic_document_student`, requires a Transformers student
profile, and uses its device and distilled token-limit defaults without
calling `create_backend()`. That profile selection, device, and checkpoint
are outside the M2b backend declaration. The graph still tracks the
catalog-derived student prompt instructions.

Each build uses the bytes it first reads for the referenced inputs. Before
registration it rereads those files once and, for M2b, rechecks the ontology
directory's YAML membership. This covers `pyproject.toml`, source files, the
catalog inputs and selected manifests, prompts, grammar, and backend YAML.
A detectable change aborts the snapshot before SQLite registration. This is
a bounded consistency check. It can miss a change reverted before its
recheck or a change after that input has been rechecked. It provides no
atomic filesystem snapshot. Node, projection, and recipe versions are
unchanged.

## M3 local execution records

A symbolic definition identifies the selected implementation and declarations.
An M3 run also identifies a completed local execution's inputs, options,
status, and materialized outputs. The existing parser and converter commands
are still available. Use the versioning commands to execute with provenance.

| Command | Required outputs | Optional outputs |
| --- | --- | --- |
| `run-parser-export` | `parser-results.jsonl`, `rejected.jsonl`, `metadata.jsonl` | `student-pairs.jsonl` with `--student-pairs` |
| `run-symbolic-export` | `documents.jsonl`, `rejected.jsonl` | `ontology-bundle.json` with `--bundle-output`, `skipped-rules.jsonl` with `--skip-unsupported-rules` |

Each command captures the current checkout's default M2b definition.
`--expected-definition ROOT_ID` can pin it to an independently selected ID.
Conversion checks the effective bundle against that definition. Parser runs
use the default catalog and active reference backend declaration and hash
the parser document exporter and its metadata implementation. The parser
wrapper accepts the original exporter's text and JSONL inputs, plus
`--corpus-id`, `--include-metta`, `--asserted-only`, and
`--require-structured-output`. It supports fresh reference exports.
Resume, custom bundles, manual backend overrides, and distilled parser
execution remain available only where supported by the original runtime.

Each version 1 run recipe uses this graph. Arrows follow dependencies.

```text
run (purpose: run)
  ├─ inputs (invocation)
  │   ├─ M2b symbolic definition
  │   ├─ raw input source
  │   ├─ effective bundle or reference backend declaration
  │   ├─ effective options
  │   └─ parser export sources (parser runs)
  ├─ outcome (status and counts)
  └─ materialized artifacts, one per output role
      └─ produced-by → same invocation
```

Input and artifact identities store raw SHA-256 and byte length under portable
logical filenames. The parser also records a hash of the effective corpus
namespace option. The registry stores neither document bodies nor credentials.
An optional output has a dependency only if the exporter materialized the
file. An empty JSONL output still has its own digest. A successful batch uses
status `completed` or `completed_with_rejections`, with counts for accepted
rows, rejected rows, and skipped rules. The CLI prints the root ID for both,
and returns 0 or 2 respectively. A fatal failure returns 1 and publishes no
run root. No timestamps enter the run identity.

The wrapper gives the producer a private copy of its input and stages outputs
under the requested destination's parent. It checks the completed artifact
set and counts, rechecks input and definition identities, then publishes
complete files using exclusive links into a new destination directory.
Leave that directory absent before capture and keep the registry elsewhere.
The staged files are cleaned up on failure. If failure occurs after filesystem
publication begins, some or all finalized files can remain in the destination.
SQLite registration is atomic. Filesystem publication and SQLite registration
are separate operations. Inputs, checkout files, and artifacts should stay
stable during capture. The bounded rechecks can miss reverted changes or
writes after a check, and there is no atomic filesystem snapshot.

### Run a local conversion and verify its manifest

The example uses a synthetic parser result and the project's optional pinned
`miner` dependency. It does not call a model service or launch a mining job.

```bash
demo=$(mktemp -d)
python - "$demo/parser-input.jsonl" <<'PY'
import sys
from pathlib import Path

from parser.ontology.catalog import load_manifest_catalog
from parser.semantic.schema import SemanticParseResult

catalog, _ = load_manifest_catalog()
text = "Mira left."
fixture = SemanticParseResult.model_validate({
    "document": {"document_id": "demo", "corpus_id": "demo", "source_text": text},
    "provenance": {
        "provider": "fixture", "model": "fixture",
        "parser_version": "1.0.0", "prompt_version": "fixture",
        "ontology_versions": list(catalog.versions),
    },
    "extraction": {
        "entities": [{"id": "e1", "entity_type": "Event", "mention": {"text": text}}],
        "assertions": [{
            "predicate": "Event", "arguments": [{"entity_id": "e1", "role": "event_id"}],
            "factuality": "asserted", "confidence": 0.95, "evidence": [{"text": text}],
        }],
        "rules": [],
    },
})
Path(sys.argv[1]).write_text(fixture.model_dump_json() + "\n", encoding="utf-8")
PY
run_id=$(python -m versioning run-symbolic-export \
    --input "$demo/parser-input.jsonl" --output-dir "$demo/converted" \
    --registry "$demo/graph.sqlite3" --bundle-output)
python -m versioning inspect --registry "$demo/graph.sqlite3" "$run_id"
python -m versioning export-manifest --registry "$demo/graph.sqlite3" "$run_id" \
    > "$demo/run-manifest.json"
python -m versioning verify-manifest "$demo/run-manifest.json" --expected-root "$run_id"
```

With a configured reference backend, run the parser on a local text file
(one document per nonempty line):

```bash
printf '%s\n' 'Mira left.' > "$demo/source.txt"
python -m versioning run-parser-export --input "$demo/source.txt" \
    --output-dir "$demo/parsed" --registry "$demo/graph.sqlite3" --corpus-id demo
```

Offline manifest verification recomputes identities and validates the graph.
Supply a separately trusted expected root to verify the graph against that
identity. Compare local file digests and lengths to artifact nodes to check
the bytes themselves. Anyone can construct a correctly hashed graph with
fabricated execution claims. Verification alone cannot establish that a
producer ran, attest resolved model weights or service state, or verify the
source behind a declared Git selector. An imported parser-result JSONL file
contributes its raw identity. Its original parser execution is unknown to
the conversion wrapper. This repository's supported M3 boundary ends at the
files handed off to Hybrid Miner. Mining output provenance requires a
supported and testable external mining execution path.

A future cache could look up completed conversions with identical invocation
IDs and verify that the materialized files still match the recorded digests
before reuse. A matching parser invocation ID identifies the same declared
inputs and options. A remote or nondeterministic backend may still produce
different bytes and a different run ID. M3 always executes the producer.

For independent review, verify the portable graph, then check its M3 recipe
and the supplied artifact files. Both APIs are read-only.

```python
from pathlib import Path
from versioning.execution import validate_symbolic_run, verify_run_artifacts
from versioning.manifest import verify_manifest

nodes = verify_manifest(Path("run.json").read_bytes(), expected_root_id=trusted_root)
validate_symbolic_run(nodes, trusted_root)
verify_run_artifacts(nodes, trusted_root, {
    "accepted": Path("converted/documents.jsonl"),
    "rejected": Path("converted/rejected.jsonl"),
})  # Supply every optional output present in this run as well.
```

The recipe check enforces the M2b definition, shared declaration and
invocation, exact artifact roles, effective options, and outcome invariants.
Matching artifact bytes support integrity only. Identical run IDs can
represent multiple observations. Publication assumes a trusted destination
parent and registry path. The wrapper anchors output links to a directory
descriptor and refuses final-component symlinks during artifact verification.
After a publication failure, inspect any finalized files and retry with a
fresh directory. Abrupt process termination may also leave private staging
files. The tracker stores completed observations only.
