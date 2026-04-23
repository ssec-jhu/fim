"""Utility helpers for FIM: package discovery and on-demand test-data fetching.

Heavy ``.npy`` and ``.inp`` fixtures are **not** committed to the repository
(only the small ``simulate/`` TIFFs ship in-tree). Instead they are published
as per-release assets on the GitHub releases page:

    https://github.com/ssec-jhu/fim/releases/download/<tag>/<dataset>.tar.gz

On first use, :func:`fetch_dataset` downloads and extracts the archive into a
user-level cache directory, verifies its SHA-256 checksum against
:data:`DATASETS`, and returns the local path. Subsequent calls are
no-ops.

Environment variables
---------------------
FIM_DATA_DIR
    Override the cache directory. Default: ``<fim package>/test_data`` — see
    :func:`data_cache_dir`. Set this (e.g. to ``~/.cache/fim``) when the
    package directory is not writable, for example in a system-wide install
    or a read-only container image.
FIM_DATA_URL_BASE
    Override the URL base. Useful for mirrors or offline re-hosting.
FIM_DATA_TAG
    Override the release tag the manifest points at.

CLI
---
``python -m fim.util list``                List known datasets.
``python -m fim.util where``               Print the active cache directory.
``python -m fim.util fetch <name> [...]``  Download one or more datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from . import __project__


# ---------------------------------------------------------------------------
# Package / repo discovery (kept backwards-compatible)
# ---------------------------------------------------------------------------
def find_package_location(package: str = __project__) -> Path:
    """Return the on-disk location of ``package``."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"Cannot locate package {package!r}")
    return Path(next(iter(spec.submodule_search_locations)))


def find_repo_location(package: str = __project__) -> Path:
    """Return the repo root (parent of the installed package)."""
    return find_package_location(package).parent


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Description of a downloadable fixture archive."""

    name: str
    archive: str  # filename of the release asset (e.g. "80um.tar.gz")
    sha256: str  # hex digest of the archive
    description: str = ""


#: Release tag on ``ssec-jhu/fim`` that hosts the dataset assets.
DEFAULT_DATA_TAG = "data-v1"

#: Base URL for dataset downloads. Override via ``FIM_DATA_URL_BASE``.
DEFAULT_DATA_URL_BASE = f"https://github.com/ssec-jhu/fim/releases/download/{DEFAULT_DATA_TAG}"

#: Known fixtures. Fill in ``sha256`` after uploading the archives.
DATASETS: dict[str, DatasetSpec] = {
    "80um": DatasetSpec(
        name="80um",
        archive="80um.tar.gz",
        sha256="fb28d3ae7854ad69e55e8213b137a49c8bafa6acc8957e291803e01b4d9ed696",
        description="Linear model default fixture (small synthetic stack).",
    ),
    "HGO": DatasetSpec(
        name="HGO",
        archive="HGO.tar.gz",
        sha256="cefa040815ac7f7bd2c52aa934743e3f92a88c8e432f64b5bf7d5fe9b57b1420",
        description="HGO model default fixture; includes 350k.inp mesh.",
    ),
    "NH": DatasetSpec(
        name="NH",
        archive="NH.tar.gz",
        sha256="f306649d50b98242470a076cff0b257dae337df7188a086e54c4a84332f2798a",
        description="Neo-Hookean model default fixture; includes 335k_32um.inp mesh.",
    ),
    "exp2-benchmark": DatasetSpec(
        name="exp2-benchmark",
        archive="exp2-benchmark.tar.gz",
        sha256="3a27be38a52526b3cc06a135bb6a2cc66b85ec3927dd17f3126d9d80ad346e4f",
        description="Experimental benchmark used by auto-tune RMSE evaluation.",
    ),
}


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
def data_cache_dir() -> Path:
    """Return the directory in which fixtures are cached.

    Priority:

    1. ``$FIM_DATA_DIR`` (expanded, resolved) — recommended for read-only
       installs (system Python, Docker images) or when you want to share the
       cache across multiple environments.
    2. ``<fim package>/test_data`` — matches the historical in-tree layout
       so default paths like ``fim/test_data/80um`` keep resolving for
       editable installs.
    """
    override = os.environ.get("FIM_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (find_package_location() / "test_data").resolve()


# ---------------------------------------------------------------------------
# Download / extract helpers
# ---------------------------------------------------------------------------
_CHUNK = 1 << 20  # 1 MiB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def _download(url: str, dst: Path) -> None:
    """Stream ``url`` to ``dst`` (atomic: writes to ``<dst>.part`` first).

    Only ``http://`` and ``https://`` URLs are accepted; other schemes
    (``file://``, ``ftp://``, …) are rejected to avoid accidentally
    reading arbitrary local files via ``$FIM_DATA_URL_BASE``.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"Refusing to fetch {url!r}: scheme {scheme!r} not in {sorted(_ALLOWED_URL_SCHEMES)}.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:  # noqa: S310  # nosec B310 - scheme validated above
            shutil.copyfileobj(resp, f, length=_CHUNK)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    tmp.replace(dst)


def _is_within(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _safe_extract_tar(tar_path: Path, dest: Path) -> None:
    """Extract ``tar_path`` into ``dest``, rejecting path-traversal entries.

    Each member is individually validated (directory confinement + regular
    file / directory only) and extracted one at a time; ``extractall`` is
    never called so this is safe on pre-3.12 runtimes that lack the
    ``filter='data'`` default.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if not (member.isreg() or member.isdir()):
                raise RuntimeError(
                    f"Unsupported tar entry {member.name!r} (type={member.type!r}); "
                    f"only regular files and directories are allowed."
                )
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise RuntimeError(f"Unsafe path in archive: {member.name!r}")
            target = dest / member.name
            if not _is_within(dest, target):  # pragma: no cover - defense-in-depth for symlinked dest
                raise RuntimeError(f"Unsafe path in archive: {member.name!r}")
            # Strip link info and world-writable bits defensively.
            member.mode = member.mode & 0o755
            tf.extract(member, dest)  # noqa: S202  # member validated: type, name, and resolved path


# ---------------------------------------------------------------------------
# Public API: fetch / resolve / list
# ---------------------------------------------------------------------------
def available_datasets() -> list[str]:
    """Return the names of all datasets known to the manifest."""
    return sorted(DATASETS)


def resolve_dataset(name: str, *, auto_fetch: bool = False) -> Path:
    """Return the local directory for dataset ``name``.

    Parameters
    ----------
    name
        One of :func:`available_datasets`.
    auto_fetch
        When ``True``, download the dataset if it is not already cached.
        When ``False`` (default) and the directory is missing, raise
        :class:`FileNotFoundError` with a hint on how to fetch it.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Known: {available_datasets()}")

    target = data_cache_dir() / name
    if target.is_dir() and any(target.iterdir()):
        return target
    if auto_fetch:
        return fetch_dataset(name)
    raise FileNotFoundError(
        f"Dataset {name!r} is not cached at {target}. "
        f"Run `python -m fim.util fetch {name}` or call "
        f"`fim.util.fetch_dataset({name!r})`."
    )


def fetch_dataset(
    name: str,
    *,
    url_base: str | None = None,
    force: bool = False,
) -> Path:
    """Download and extract dataset ``name``; return its local directory.

    The archive is verified against the SHA-256 recorded in :data:`DATASETS`
    before extraction. Existing data is reused unless ``force=True``.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Known: {available_datasets()}")

    spec = DATASETS[name]
    cache_root = data_cache_dir()
    target = cache_root / name
    if target.is_dir() and any(target.iterdir()) and not force:
        return target

    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cache directory {cache_root} is not writable ({exc}). "
            f"Set FIM_DATA_DIR to a writable path (e.g. ~/.cache/fim) and retry."
        ) from exc

    base = (url_base or os.environ.get("FIM_DATA_URL_BASE") or DEFAULT_DATA_URL_BASE).rstrip("/")
    url = f"{base}/{spec.archive}"

    with tempfile.TemporaryDirectory(prefix="fim-dl-") as tmpdir:
        archive = Path(tmpdir) / spec.archive
        print(f"[fim] downloading {url}", file=sys.stderr)
        _download(url, archive)

        if spec.sha256:
            actual = _sha256(archive)
            if actual.lower() != spec.sha256.lower():
                raise RuntimeError(f"Checksum mismatch for {spec.archive}: expected {spec.sha256}, got {actual}")
        else:
            print(
                f"[fim] warning: no SHA-256 recorded for {name!r}; skipping verification",
                file=sys.stderr,
            )

        if force and target.exists():
            shutil.rmtree(target)
        _safe_extract_tar(archive, target.parent)

    if not target.is_dir():
        # Tarball didn't contain a top-level <name>/ directory; wrap what we got.
        raise RuntimeError(f"Archive {spec.archive} did not extract a {name!r} directory into {target.parent}.")
    return target


def fetch_datasets(names: Iterable[str], *, force: bool = False) -> list[Path]:
    """Fetch multiple datasets; return their local paths in input order."""
    return [fetch_dataset(n, force=force) for n in names]


# ---------------------------------------------------------------------------
# CLI entry point: ``python -m fim.util ...``
# ---------------------------------------------------------------------------
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fim.util", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List known datasets.")
    sub.add_parser("where", help="Print the active cache directory.")

    pf = sub.add_parser("fetch", help="Download one or more datasets.")
    pf.add_argument("names", nargs="+", choices=available_datasets(), metavar="NAME")
    pf.add_argument("--force", action="store_true", help="Re-download even if cached.")
    pf.add_argument("--url-base", default=None, help="Override the base URL.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)

    if args.cmd == "list":
        for name in available_datasets():
            spec = DATASETS[name]
            print(f"{name:16s}  {spec.archive:28s}  {spec.description}")
        return 0

    if args.cmd == "where":
        print(data_cache_dir())
        return 0

    if args.cmd == "fetch":
        for name in args.names:
            path = fetch_dataset(name, url_base=args.url_base, force=args.force)
            print(path)
        return 0

    return 2  # pragma: no cover - argparse rejects unknown subcommands before we get here


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
