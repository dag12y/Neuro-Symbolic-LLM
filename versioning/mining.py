"""Source-selected execution of the five CLIs documented by hybrid-miner v0.1.1."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from .nodes import Node
from .symbolic import read_hybrid_miner_release
from .workflow import PROJECT, composite, observation


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repo), *args],
        capture_output=True,
        timeout=30,
    )
    if result.returncode:
        raise ValueError(f"Git observation unavailable ({args[0]})")
    return result.stdout


def release(repo: Path | None, remote: bool = False) -> tuple[Node, ...]:
    declared = read_hybrid_miner_release(PROJECT / "pyproject.toml")
    selector = cast(Mapping[str, str], declared.payload["selector"])
    ref = ("refs/tags/" if selector["kind"] == "tag" else "") + selector["value"]
    commit = None
    method = "unavailable"
    if repo is not None:
        url = git(repo, "remote", "get-url", "origin").decode().strip()
        if url.lower().removesuffix(".git") != str(
            declared.payload["repository"]
        ).lower().removesuffix(".git"):
            raise ValueError("Miner source repository differs from declaration")
        commit = (
            git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
        )
        method = "local-git"
    elif remote:
        # A peeled ref can name a tree or blob. Require Git to establish the
        # object type before writing a field named commit.
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            scratch = Path(directory)
            try:
                git(scratch, "init", "--bare", "--quiet")
                git(
                    scratch,
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    "--no-tags",
                    "--filter=blob:none",
                    str(declared.payload["repository"]),
                    ref,
                )
                commit = (
                    git(scratch, "rev-parse", "--verify", "FETCH_HEAD^{commit}")
                    .decode()
                    .strip()
                )
            except (ValueError, OSError, subprocess.TimeoutExpired):
                pass
            else:
                method = "remote-ref"
    obs = observation("workflow/miner/resolution", {"method": method, "commit": commit})
    root = composite(
        "workflow/miner/release",
        "definition",
        {"declaration": declared, "resolution": obs},
    )
    return declared, obs, root


def validate_release(nodes: tuple[Node, ...]) -> None:
    from .workflow import slots

    root = nodes[-1]
    edges = slots(root, {n.id: n for n in nodes})
    if set(edges) != {"declaration", "resolution"}:
        raise ValueError("Invalid release observation")
    decl, obs = edges["declaration"], edges["resolution"]
    value = cast(dict[str, Any], obs.to_dict()["payload"]).get("value", {})
    commit = value.get("commit")
    if (
        decl.kind != "external_release"
        or decl.name != "hybrid-miner"
        or str(decl.payload["repository"]).lower()
        != "https://github.com/icog-labs-dev/hybrid-miner.git"
        or obs != observation("workflow/miner/resolution", value)
        or set(value) != {"method", "commit"}
        or value["method"] not in ("unavailable", "remote-ref", "local-git")
        or (commit is None) != (value["method"] == "unavailable")
        or (
            commit is not None
            and (
                not isinstance(commit, str)
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
            )
        )
        or root != composite("workflow/miner/release", "definition", edges)
    ):
        raise ValueError("Invalid miner release identity")
    selector = cast(Mapping[str, str], decl.payload["selector"])
    if commit and selector["kind"] == "commit" and commit != selector["value"]:
        raise ValueError("Resolved commit differs from selector")


def materialize(repo: Path, commit: str, directory: Path) -> str:
    """Load regular package blobs from the selected Git tree into a fresh import root."""
    listing = git(repo, "ls-tree", "-rz", commit, "src/hybrid_miner")
    for record in listing.rstrip(b"\0").split(b"\0"):
        header, path = record.split(b"\t", 1)
        mode, kind, oid = header.split()
        relative = Path(path.decode()).relative_to("src")
        if (
            mode not in (b"100644", b"100755")
            or kind != b"blob"
            or ".." in relative.parts
            or relative.suffix == ".pyc"
        ):
            raise ValueError("Unsupported miner Git tree entry")
        content = git(repo, "cat-file", "blob", oid.decode())
        blob = b"blob " + str(len(content)).encode() + b"\0" + content
        if (
            hashlib.new("sha1" if len(oid) == 40 else "sha256", blob)
            .hexdigest()
            .encode()
            != oid
        ):
            raise ValueError("Git blob verification failed")
        output = directory / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
    if not (directory / "hybrid_miner/__init__.py").is_file():
        raise ValueError("Selected source lacks the Hybrid Miner package")
    return package_digest(directory / "hybrid_miner")


def mine(
    private: Path,
    out: Path,
    runtime: dict[str, Any],
    *,
    repo: Path,
    commit: str,
    petta: Path,
    miner: Path,
    support: int,
) -> None:
    source = private.parent / "miner-source"
    digest = materialize(repo, commit, source)
    # -I excludes PYTHONPATH, user site and the working directory. The verified
    # regular package wins module selection over any installed distribution.
    bootstrap = (
        "import pathlib,sys,runpy;root=pathlib.Path(sys.argv.pop(1));sys.path.insert(0,str(root));"
        "import hybrid_miner;assert pathlib.Path(hybrid_miner.__file__)==root/'hybrid_miner/__init__.py';"
        "m=sys.argv.pop(1);runpy.run_module(m,run_name='__main__',alter_sys=True)"
    )
    common = ["--bundle", str(private / "bundle.json")]
    engine = [
        "--petta-root",
        str(petta.resolve()),
        "--miner-root",
        str(miner.resolve()),
        "--min-support",
        str(support),
    ]
    documents = ["--documents", "accepted.jsonl"]
    steps = [
        (
            "ingestion",
            [
                *common,
                "--input",
                str(private / "documents.jsonl"),
                "--accepted-output",
                "accepted.jsonl",
                "--rejected-output",
                "rejected.jsonl",
            ],
        ),
        (
            "emission",
            [
                *common,
                "--input",
                "accepted.jsonl",
                "--atoms-output",
                "facts.metta",
                "--metadata-output",
                "metadata.metta",
                "--provenance-output",
                "provenance.jsonl",
            ],
        ),
        (
            "mining.admitted_support",
            [
                *common,
                *documents,
                *engine,
                "--emitted-atoms",
                "facts.metta",
                "--metadata",
                "metadata.metta",
                "--provenance",
                "provenance.jsonl",
                "--admitted-output",
                "admitted.metta",
                "--admission-output",
                "admission.json",
                "--quarantine-output",
                "quarantine.jsonl",
                "--support-output",
                "support.jsonl",
            ],
        ),
        (
            "mining.exact_matches",
            [
                *common,
                *documents,
                *engine,
                "--atoms",
                "admitted.metta",
                "--provenance",
                "provenance.jsonl",
                "--admission",
                "admission.json",
                "--support",
                "support.jsonl",
                "--output",
                "exact.jsonl",
                "--rejected-output",
                "exact-rejected.jsonl",
            ],
        ),
        (
            "mining.build_pattern_records",
            [
                *common,
                *documents,
                "--atoms",
                "admitted.metta",
                "--provenance",
                "provenance.jsonl",
                "--admission",
                "admission.json",
                "--exact",
                "exact.jsonl",
                "--output",
                "patterns.jsonl",
                "--rejected-output",
                "patterns-rejected.jsonl",
            ],
        ),
    ]
    engine_heads = {}
    for name, checkout in (("petta", petta), ("hyperon_miner", miner)):
        try:
            if (
                Path(
                    git(checkout, "rev-parse", "--show-toplevel").decode().strip()
                ).resolve()
                != checkout.resolve()
            ):
                raise ValueError("Expected a Git checkout root")
            head = git(checkout, "rev-parse", "HEAD").decode().strip()
        except (OSError, ValueError):
            head = None
        engine_heads[name] = head
    runtime.update(
        commit=commit,
        package_tree_sha256=digest,
        engine_heads_reported=engine_heads,
        steps=[],
    )
    for module, args in steps:
        code = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                bootstrap,
                str(source),
                "hybrid_miner." + module,
                *args,
            ],
            cwd=out,
            stdout=2,
            timeout=3600,
        ).returncode
        runtime["steps"].append({"module": module, "code": code})
        if code and not (module == "ingestion" and code == 2):
            raise ValueError(f"Miner {module} exited {code}")
    if digest != package_digest(source / "hybrid_miner"):
        raise ValueError("Miner source changed during execution")


def package_digest(package: Path) -> str:
    """Digest regular source/resource bytes in path order, excluding interpreter caches."""
    from .execution import _file_identity

    inventory = hashlib.sha256()
    for path in sorted(
        package.rglob("*"), key=lambda p: p.relative_to(package).as_posix()
    ):
        rel = path.relative_to(package)
        if "__pycache__" in rel.parts:
            continue
        if path.is_symlink():
            raise ValueError("Symlink in package inventory")
        if path.is_file():
            identity = _file_identity(path, follow_links=False)
            inventory.update(
                b"src/hybrid_miner/"
                + rel.as_posix().encode()
                + b"\0"
                + bytes.fromhex(str(identity["sha256"]))
            )
    return inventory.hexdigest()


def _comparison(expected: str, installed: str | None) -> str:
    if installed is None:
        return "unavailable"
    return "match" if installed == expected else "mismatch"


def installed_release(repo: Path) -> tuple[Node, ...]:
    from importlib.metadata import PackageNotFoundError, distribution
    from tempfile import TemporaryDirectory

    resolved = release(repo)
    current = cast(dict[str, Any], resolved[1].to_dict()["payload"])["value"]["commit"]
    with TemporaryDirectory() as tmp:
        expected = materialize(repo, current, Path(tmp))
    installed = None
    version = None
    try:
        dist = distribution("hybrid-miner")
        version = dist.version
        package = Path(str(dist.locate_file("hybrid_miner")))
        if package.is_dir() and not package.is_symlink():
            installed = package_digest(package)
    except PackageNotFoundError:
        pass
    result = observation(
        "workflow/miner/installed-tree",
        {
            "version": version,
            "source_sha256": expected,
            "installed_sha256": installed,
            "comparison": _comparison(expected, installed),
        },
    )
    root = composite(
        "workflow/miner/installation",
        "component",
        {"release": resolved[-1], "inspection": result},
    )
    return *resolved, result, root


def validate_installation(node: Node, index: Mapping[str, Node]) -> None:
    from .nodes import validate_id
    from .workflow import slots

    edges = slots(node, index)
    if (
        set(edges) != {"release", "inspection"}
        or edges["release"].name != "workflow/miner/release"
        or node != composite("workflow/miner/installation", "component", edges)
    ):
        raise ValueError("Invalid miner installation record")
    obs = edges["inspection"]
    value = cast(dict[str, Any], obs.to_dict()["payload"]).get("value", {})
    if set(value) != {
        "version",
        "source_sha256",
        "installed_sha256",
        "comparison",
    } or obs != observation("workflow/miner/installed-tree", value):
        raise ValueError("Invalid installed package observation")
    validate_id(value["source_sha256"])
    if value["installed_sha256"] is not None:
        validate_id(value["installed_sha256"])
        if not isinstance(value["version"], str):
            raise ValueError("Missing installed version")
    comparison = _comparison(value["source_sha256"], value["installed_sha256"])
    if value["comparison"] != comparison:
        raise ValueError("Inconsistent installation comparison")
