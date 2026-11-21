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

An edge contains exactly `role`, `slot`, and `target_id`. The roles
`uses-source`, `uses-release`, and `uses-component` require targets of kinds
`source`, `external_release`, and `composite`, respectively. Slots are logical
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
