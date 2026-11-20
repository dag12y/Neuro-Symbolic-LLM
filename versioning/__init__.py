"""Content-addressed component provenance and offline manifests."""

from .compatibility import CompatibilityResult, CompatibilityStatus
from .hashing import canonical_json, hash_directory, hash_file, hash_identity, load_json
from .manifest import export_manifest, verify_manifest
from .records import ComponentRecord, Dependency, LabelRole
from .registry import Registry

__all__ = [
    "CompatibilityResult",
    "CompatibilityStatus",
    "ComponentRecord",
    "Dependency",
    "LabelRole",
    "Registry",
    "canonical_json",
    "export_manifest",
    "hash_directory",
    "hash_file",
    "hash_identity",
    "load_json",
    "verify_manifest",
]
