# Core version tracker

The tracker gives local components a shared provenance format. A component record
captures what was identified, how its content was hashed, who produced it when
known, and which exact records it depends on. The core runs with the Python 3.11+
standard library and requires no Git checkout.

## Records and identity

A label is a human-readable name. A release label can identify only one release
record for a given kind and name within a registry. Family and locator labels
can appear on multiple records. Use the full record ID for exact references.
A record without a label uses the `none` role.

Each record has schema and canonicalization versions, a kind and name, a SHA-256
content digest, and a caller-defined digest scope describing exactly what was
hashed, such as `raw file bytes` or `directory tree v1`. A dependency specifies
a role such as `schema` and the target record ID. Dependencies are ordered.
Optional producer information and details are JSON objects. Records freeze
nested objects and arrays, and `to_dict()` returns a mutable copy.

Every persisted field other than `record_id` participates in record identity,
including the label, digest scope, dependencies, producer, and details. Prefer
stable metadata. If you deliberately include a machine path or timestamp in
the record, changing that value changes the record ID.

## Canonicalization v1

`canonical_json` emits ASCII JSON bytes with Python's `json.dumps` encoding,
sorted keys in Unicode code-point order, separators `,` and `:`, ASCII escapes,
and no trailing newline. Arrays retain their order. Strings are used as
supplied, without Unicode normalization. Surrogate code points are rejected. Numbers follow Python's JSON encoder
for integers and finite binary64 floats, including the distinction between
`1` and `1.0`. Objects require string keys. `load_json` rejects duplicate keys
at every level as well as nonfinite values.

`hash_identity` hashes the prefix `versioning:component:v1` followed by a NUL
byte and the canonical payload. `hash_file` streams raw file bytes. Directory
hashing uses the prefix `versioning:directory:v1` followed by a NUL byte and a
canonical array of entries sorted by relative path. Each entry is
`[path, "directory"]` or `[path, "file", file_sha256]`. Paths use slash-separated
relative names. Empty directories affect the digest. The root path, timestamps,
and file permissions are excluded. All symlinks and root paths containing
`..` are rejected. Hash a stable tree to obtain a consistent snapshot.

## Registry and manifests

Register dependencies first, then register their dependents with
`Registry.register()`. Registering the same record again returns its ID.
`get(id)` verifies the record against its contents and filename. `find()`
filters by kind, name, and optionally label. Registration checks dependency
closure and release labels under a writer lock, then publishes a complete file
with a no-overwrite hard link. M1 uses a local POSIX filesystem with support
for advisory locks and hard links. Searches and registration scan record files,
which keeps the implementation small for lightweight use.

`export_manifest()` returns a versioned JSON document containing the root ID
and all reachable records. `verify_manifest()` checks the schema, each record
ID, dependency closure, and release-label consistency offline. It returns the
verified records in dependency order, with the root last.

Manifests attest to internal consistency. Authenticity requires a trusted
root ID or an external trust mechanism. Artifact bytes are external and must
be checked separately against the declared digests and scopes. Compatibility
is caller-defined and can be reported as `compatible`, `incompatible`, or
`unknown`. The tracker assigns no compatibility verdict based on hashes.

## Executable example

Run from the repository root.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from versioning import ComponentRecord, Registry, export_manifest, hash_file, verify_manifest

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    prompt = root / "prompt.txt"
    prompt.write_bytes(b"Extract facts.")
    registry = Registry(root / ".version-registry")
    record_id = registry.register(
        ComponentRecord(
            kind="prompt",
            name="extract",
            label="v1",
            label_role="family",
            content_digest=hash_file(prompt),
            digest_scope="raw file bytes",
        )
    )
    found = registry.get(record_id)
    print(record_id)
    print(found.kind, found.name, found.label)
    assert verify_manifest(export_manifest(registry, record_id))[-1] == found
```

## Register the repository's semantic parser definitions

Run from the checkout root with the project's Python environment. The adapter
uses the existing PyYAML-based loaders. It registers the modules selected by
the default `semantic-extraction-v1.yaml` bundle and checks the bundle's version
pins before registering any records. Ontology modules and the bundle use their
declared release labels and raw YAML byte digests. A byte change, including a
comment-only edit, conflicts with an existing release label in the same
registry. Semantic compatibility remains a separate decision.

The contract record uses the exact bytes of `parser/semantic/schema.py` and its
declared `SEMANTIC_CONTRACT_VERSION`. The parser implementation record uses
`PARSER_VERSION` and hashes the parser, backend adapters, ontology loader,
normalization, output cleanup, MeTTa rendering, and Atomese grammar sources.
Both labels are locators. The implementation record links to the contract,
bundle, and prompt records. Set `include_git_provenance=True` to attach the
checkout's HEAD commit when it contains the exact captured implementation
source files. This optional producer field affects the implementation record
ID. Copies without Git metadata can still be registered.

The adapter records router, general, discourse, routed extraction, and student
prompt definitions. The three teacher YAML labels and the student label are
families. Routed extraction links the general and discourse prompts and has no
separate label in the parser. It includes the shared Python source containing
the combined instructions and framing. Teacher definitions use the full
`catalog_prompt.py` source, so an edit there can change multiple teacher IDs.
The student definition uses `student_prompt.py`. Source-set digests hash
`versioning:source-set:v1` followed by a NUL byte and the canonical mapping from
repository-relative paths to raw file SHA-256 digests. Prompt dependencies
cover the contract and the ontology inputs used by each definition.
The router reads no ontology guide. Runtime prompt-version and fingerprint
strings remain in the parser's existing APIs.

This snapshot covers the default bundle and selected source definitions. It
does not capture the final provider request, document or context text, selected
runtime route, correction message instances, tokenizer chat templates, model
weights, generation settings, or installed dependency versions. Each record
is published individually. If a release conflict interrupts registration,
earlier successful records are already available.

The following commands register into a caller-selected directory, inspect
records by ID or label, and write a portable manifest for manual verification.

```bash
REGISTRY_DIR=.version-registry/semantic-demo
MANIFEST_PATH=/tmp/semantic-components.json

.venv/bin/python - "$REGISTRY_DIR" "$MANIFEST_PATH" <<'PY'
import sys
from pathlib import Path

from versioning import Registry, export_manifest
from versioning.semantic import register_semantic_components

registry = Registry(sys.argv[1])
ids = register_semantic_components(registry, project_root=Path.cwd())

print("implementation", ids.parser)
print("contract", ids.contract)
print("bundle", ids.ontology_bundle)
print("modules", dict(ids.ontology_modules))
print("prompts", dict(ids.prompts))
print(registry.get(ids.contract).to_dict())
for record in registry.find("teacher-prompt", "general", label="general-v1"):
    print(record.to_dict())

Path(sys.argv[2]).write_bytes(export_manifest(registry, ids.parser))
PY

.venv/bin/python - "$MANIFEST_PATH" <<'PY'
import sys
from pathlib import Path

from versioning import verify_manifest

records = verify_manifest(Path(sys.argv[1]).read_bytes())
print("verified", len(records), "records with root", records[-1].record_id)
PY
```

## Version manifests for parser exports

Run the normal export with an already configured model profile and a new
output path. This command runs the parser against your configured backend.

```bash
.venv/bin/python -m parser.semantic export-documents \
  --input data/source.jsonl --output data/results.jsonl
```

The exporter writes `data/results.jsonl.versions.json` beside the finalized
parse-result JSONL. Python callers receive its path as
`ExportResult.versions_path`. The sidecar covers the primary JSONL. Rejected
rows, source metadata, and optional student pairs keep their existing formats
and are outside this sidecar's verification scope.

The JSON has three top-level fields.

| Field | Contents |
| --- | --- |
| `export_manifest_version` | Format version `1` |
| `artifact` | M1 `parser-export-artifact` record. Its `name` is the portable output basename, its `content_digest` hashes the exact final JSONL bytes, and its single dependency is the parser implementation record. `details` holds the input file SHA-256, allowlisted effective settings, and unresolved items. |
| `components` | Embedded M1 manifest with the parser implementation as its root and the complete reachable component records, including available version labels. |

An effective catalog that matches the validated default catalog uses the M2
bundle records. Other catalogs use a hash of the actual in-memory definitions
and report their declared ontology versions. Their original YAML files and
bundle labels are unresolved. Run settings include routing, structured-output
and rendering options, and available numeric or boolean backend settings.
The general guidance hash is the baseline fingerprint exposed by the parser.
Per-document prompt fingerprints remain in each exported row's provenance.
The sidecar excludes document text, input paths, and arbitrary model and
provider strings.

### Inspect and verify offline

```bash
.venv/bin/python -m json.tool data/results.jsonl.versions.json

.venv/bin/python - data/results.jsonl.versions.json <<'PY'
import sys

from versioning.export import verify_export_sidecar

artifact = verify_export_sidecar(sys.argv[1])
print("Verified artifact ID:", artifact.record_id)
print("Output SHA-256:", artifact.content_digest)
print("Parser record ID:", artifact.dependencies[0].record_id)
PY
```

Copy the JSONL and sidecar together, keeping their basenames. Verification
checks the embedded component graph, the artifact record and its parser link,
and the adjacent output bytes. It reports malformed or inconsistent data and
missing files. This checks internal integrity. Authenticity requires a
trusted artifact ID or an external signature. The recorded source hash can be
compared separately when the original input is available.

### Recovery

The sidecar is published from a temporary file after all output files are
finalized and before the export checkpoint is removed. A sidecar publication
failure reports an error and retains the complete checkpoint. Fix the cause
and rerun the same command with `--resume` to finish publication without
repeating inference. A resume also completes any pending artifact renames.
If publication succeeded before an interruption in checkpoint cleanup, resume
accepts the identical sidecar and removes the checkpoint.

A fresh run refuses existing output, partials, checkpoints, and sidecars.
For a fresh export, choose a new output path or move the old output set and
its sidecar together first. Resume refuses a sidecar left beside an incomplete
export or a conflicting sidecar beside a complete one. Inspect and move the
stale sidecar aside before retrying. Use one writer per export destination.

New checkpoints pin the captured component identity and settings. Pre-M3
checkpoints can resume under the existing fingerprint checks, with the
earlier prefix's exact component provenance reported as unresolved. Keep the
source input, checkout, and backend setup stable during an export and its
recovery. If finalization detects an input or definition change during the
run, the checkpoint is invalidated and a fresh export is required. The manifest records selected source definitions and effective
options. It cannot reproduce fully rendered provider requests, provider-side
state, resolved model weights, tokenizer templates, or the exact dependency
environment.

## Student training and checkpoints

The student trainer saves one final Hugging Face checkpoint after the last
validation pass and, for LoRA, after merging adapters. Model files, tokenizer
files, and the existing `training_metadata.json` live in the checkpoint
directory. A fixed tracking flag inside the directory indicates that a sidecar
is expected. A sibling file named `<directory>.checkpoint.versions.json` travels
with the directory. It contains `checkpoint_manifest_version: 1` and a portable
M1 manifest in `components`, rooted at a `student-checkpoint` record.

The checkpoint digest uses the M1 directory format. It covers all relative
names, regular file bytes, shards, metadata, and empty directories. Use a
dedicated checkpoint directory. Adding an unrelated file also changes its
digest. The digest includes the fixed flag. The sibling sidecar and its saving
marker live outside the hashed directory. The `training` dependency identifies a `student-training` record. Its
two dependencies identify the student prompt and the effective ontology
bundle or in-memory catalog, including the reachable contract and modules.

The training record contains raw JSONL SHA-256 digests computed while reading
and separate ordered digests for the actual training and validation token
examples. Each token digest streams canonical JSON examples followed by LF
before shuffling or batch padding. The record also includes the seed, split
mode, effective training settings, parsed config digest, selected training
source hashes, Python and library versions, and the CUDA build version.
LoRA module selections are hashed. Local model paths and free-form locators
are represented by a digest. A namespaced model identifier is included when the loaded model configuration
reports a resolved commit and the identifier does not point to a local path. Nulls
indicate unavailable values. Uncaptured weights, tokenizer identity, runtime
state and other reproducibility gaps are listed under `unresolved`.
Raw examples and arbitrary config fields stay out of the manifest. The
existing Hugging Face artifacts and training metadata retain their own content.

### Inspect and verify

From the repository root, set the directory to the trained checkpoint.

```bash
CHECKPOINT=models/semantic_document_parser
.venv/bin/python -m json.tool "${CHECKPOINT}.checkpoint.versions.json"
.venv/bin/python - "$CHECKPOINT" <<'PY'
import sys
from versioning.checkpoint import verify_student_checkpoint

record = verify_student_checkpoint(sys.argv[1])
if record is None:
    print("Checkpoint provenance unavailable (no manifest)")
else:
    print("Checkpoint record", record.record_id)
    print("Directory SHA-256", record.content_digest)
    print("Training record", record.dependencies[0].record_id if record.dependencies else "unavailable")
PY
```

Run the offline save, overwrite, tampering and compatibility tests with
`.venv/bin/python -m pytest -q tests/unit/test_student_checkpoint_versions.py tests/unit/test_student_train.py`.
The verifier checks the directory and dependency graph without importing model
libraries or contacting a provider. Authenticity requires a separately trusted
record ID. Byte integrity does not establish model quality or deterministic
training replay.

### Interrupted saves and legacy checkpoints

Saves and verifications coordinate through a POSIX directory lock. Before the
existing in-place save starts, the writer atomically publishes a `saving`
marker at the sidecar path. It removes the previous training metadata before
writing weights and restores metadata after model and tokenizer saves succeed.
It retires the previous default weight files, shards and indexes so the loader
selects the newly saved weights.
This prevents the student loader from accepting old metadata after a partial
save. The writer replaces the marker with the success manifest only after
model, tokenizer and metadata writes finish and the directory
hash succeeds. A failure propagates to the caller. While the marker is present,
verification reports an incomplete save. Retry the full save with the trained
model still in memory, rerun training, or restore a previously verified
directory and its matching sidecar together. An interrupted in-place overwrite
may have modified the old weights. A stale or malformed sidecar blocks a new
save at that location. Archive the existing pair or choose a fresh destination.

Existing checkpoint loaders retain their contract, prompt and ontology checks.
A directory without a sidecar or tracking flag has unavailable provenance and
remains eligible for those checks. A tracked directory with a missing sidecar
fails verification, including after an interrupted save. A direct `save_model` call without a captured training
run records checkpoint integrity and explicitly marks training provenance
unavailable.
