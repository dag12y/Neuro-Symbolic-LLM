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
specific composite recipe slots and shared inputs.

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
