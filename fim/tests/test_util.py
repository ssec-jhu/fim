"""Unit tests for :mod:`fim.util`, including the dataset fetcher.

The fetcher is exercised end-to-end against a local HTTP server that serves
a tiny in-test tarball, so the tests do not depend on the network or on the
published ``data-v1`` release assets.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import socketserver
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from fim import util


# ---------------------------------------------------------------------------
# Helpers: build a fake dataset tarball in memory
# ---------------------------------------------------------------------------
def _make_tarball(name: str, files: dict[str, bytes]) -> bytes:
    """Return the bytes of a gzip tarball containing ``<name>/<path>`` entries."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, payload in files.items():
            info = tarfile.TarInfo(name=f"{name}/{rel}")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fake_release(tmp_path: Path) -> Iterator[tuple[str, bytes, str]]:
    """Spin up an HTTP server that serves a fake ``toy.tar.gz`` from ``tmp_path``.

    Yields ``(url_base, payload_bytes, sha256)``.
    """
    archive = _make_tarball("toy", {"hello.txt": b"hi\n", "data.bin": b"\x00\x01\x02"})
    (tmp_path / "toy.tar.gz").write_bytes(archive)

    handler = http.server.SimpleHTTPRequestHandler

    class _Server(socketserver.TCPServer):
        allow_reuse_address = True

    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with _Server(("127.0.0.1", 0), handler) as srv:
            port = srv.server_address[1]
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{port}", archive, _sha256(archive)
            finally:
                srv.shutdown()
                thread.join(timeout=2)
    finally:
        os.chdir(cwd)


@pytest.fixture
def patched_manifest(monkeypatch: pytest.MonkeyPatch) -> util.DatasetSpec:
    """Register a single ``toy`` dataset in :data:`util.DATASETS`."""
    spec = util.DatasetSpec(
        name="toy",
        archive="toy.tar.gz",
        sha256="",  # set per-test
        description="Tarball used by the fim.util test suite.",
    )
    monkeypatch.setitem(util.DATASETS, "toy", spec)
    return spec


@pytest.fixture
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect ``data_cache_dir`` to a tmp path via the FIM_DATA_DIR env var."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("FIM_DATA_DIR", str(cache))
    return cache


# ---------------------------------------------------------------------------
# Package discovery
# ---------------------------------------------------------------------------
def test_find_package_location_returns_fim_dir() -> None:
    path = util.find_package_location()
    assert path.is_dir()
    assert path.name == "fim"
    assert (path / "__init__.py").exists()


def test_find_repo_location_is_parent_of_package() -> None:
    assert util.find_repo_location() == util.find_package_location().parent


def test_find_package_location_unknown_package_raises() -> None:
    with pytest.raises(ModuleNotFoundError):
        util.find_package_location("definitely_not_a_real_package_xyz")


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
def test_data_cache_dir_defaults_to_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIM_DATA_DIR", raising=False)
    assert util.data_cache_dir() == (util.find_package_location() / "test_data").resolve()


def test_data_cache_dir_honours_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FIM_DATA_DIR", str(tmp_path))
    assert util.data_cache_dir() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Manifest introspection
# ---------------------------------------------------------------------------
def test_available_datasets_contains_expected_entries() -> None:
    names = set(util.available_datasets())
    assert {"80um", "HGO", "NH", "exp2-benchmark"}.issubset(names)


def test_resolve_dataset_unknown_raises() -> None:
    with pytest.raises(KeyError):
        util.resolve_dataset("does-not-exist")


def test_resolve_dataset_missing_hints_at_fetcher(isolated_cache: Path, patched_manifest: util.DatasetSpec) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        util.resolve_dataset("toy")
    assert "fetch_dataset" in str(excinfo.value)


# ---------------------------------------------------------------------------
# fetch_dataset end-to-end (against a local HTTP server)
# ---------------------------------------------------------------------------
def test_fetch_dataset_downloads_and_extracts(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
) -> None:
    url_base, _archive, sha = fake_release
    util.DATASETS["toy"] = util.DatasetSpec(**{**patched_manifest.__dict__, "sha256": sha})

    path = util.fetch_dataset("toy", url_base=url_base)

    assert path == isolated_cache / "toy"
    assert (path / "hello.txt").read_bytes() == b"hi\n"
    assert (path / "data.bin").read_bytes() == b"\x00\x01\x02"


def test_fetch_dataset_rejects_bad_checksum(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
) -> None:
    url_base, _archive, _sha = fake_release
    util.DATASETS["toy"] = util.DatasetSpec(**{**patched_manifest.__dict__, "sha256": "0" * 64})

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        util.fetch_dataset("toy", url_base=url_base)

    # Nothing partial should be left in the cache.
    assert not (isolated_cache / "toy").exists()


def test_fetch_dataset_is_noop_when_cached(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
) -> None:
    url_base, _archive, sha = fake_release
    util.DATASETS["toy"] = util.DatasetSpec(**{**patched_manifest.__dict__, "sha256": sha})

    first = util.fetch_dataset("toy", url_base=url_base)
    (first / "marker").write_text("keep-me")

    # A second call must not re-download or clobber the cached directory.
    second = util.fetch_dataset("toy", url_base="http://127.0.0.1:1/should-not-be-called")
    assert second == first
    assert (second / "marker").read_text() == "keep-me"


def test_resolve_dataset_auto_fetch(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
) -> None:
    url_base, _archive, sha = fake_release
    util.DATASETS["toy"] = util.DatasetSpec(**{**patched_manifest.__dict__, "sha256": sha})

    # Point the default base URL at the test server so resolve_dataset's
    # internal fetch_dataset call hits the local HTTP server instead of GitHub.
    import os

    os.environ["FIM_DATA_URL_BASE"] = url_base
    try:
        path = util.resolve_dataset("toy", auto_fetch=True)
    finally:
        os.environ.pop("FIM_DATA_URL_BASE", None)

    assert path.is_dir()
    assert (path / "hello.txt").exists()


# ---------------------------------------------------------------------------
# Path-traversal guard
# ---------------------------------------------------------------------------
def test_safe_extract_rejects_escape(tmp_path: Path) -> None:
    bad = tmp_path / "evil.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="../escaped.txt")
        payload = b"pwned"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    bad.write_bytes(buf.getvalue())

    dest = tmp_path / "dest"
    with pytest.raises(RuntimeError, match="Unsafe path"):
        util._safe_extract_tar(bad, dest)


def test_safe_extract_rejects_absolute_path(tmp_path: Path) -> None:
    bad = tmp_path / "abs.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="/etc/passwd")
        payload = b"x"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    bad.write_bytes(buf.getvalue())

    with pytest.raises(RuntimeError, match="Unsafe path"):
        util._safe_extract_tar(bad, tmp_path / "dest")


def test_safe_extract_rejects_non_regular_members(tmp_path: Path) -> None:
    bad = tmp_path / "sym.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        tf.addfile(info)
    bad.write_bytes(buf.getvalue())

    with pytest.raises(RuntimeError, match="Unsupported tar entry"):
        util._safe_extract_tar(bad, tmp_path / "dest")


def test_is_within_detects_escape(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    assert util._is_within(parent, parent / "child") is True
    assert util._is_within(parent, sibling) is False


# ---------------------------------------------------------------------------
# _download: URL scheme validation and network-error wrapping
# ---------------------------------------------------------------------------
def test_download_rejects_non_http_scheme(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scheme 'file'"):
        util._download("file:///etc/passwd", tmp_path / "out.bin")


def test_download_wraps_url_error(tmp_path: Path) -> None:
    # Port 1 is reserved and ~always refuses connections; the URLError
    # should be converted to a RuntimeError with a helpful message.
    dst = tmp_path / "out.bin"
    with pytest.raises(RuntimeError, match="Failed to download"):
        util._download("http://127.0.0.1:1/nope", dst)
    # Partial file must not be left behind.
    assert not dst.exists()
    assert not dst.with_name(dst.name + ".part").exists()


# ---------------------------------------------------------------------------
# resolve_dataset: cached-return branch
# ---------------------------------------------------------------------------
def test_resolve_dataset_returns_cached_target(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
) -> None:
    target = isolated_cache / "toy"
    target.mkdir(parents=True)
    (target / "marker").write_text("ok")

    assert util.resolve_dataset("toy") == target


# ---------------------------------------------------------------------------
# fetch_dataset: remaining branches
# ---------------------------------------------------------------------------
def test_fetch_dataset_unknown_raises() -> None:
    with pytest.raises(KeyError):
        util.fetch_dataset("definitely-not-a-real-dataset")


def test_fetch_dataset_unwritable_cache_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    patched_manifest: util.DatasetSpec,
) -> None:
    # A regular file sitting where the cache dir should be forces mkdir() to
    # raise OSError, which fetch_dataset wraps in a helpful RuntimeError.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setenv("FIM_DATA_DIR", str(blocker))

    with pytest.raises(RuntimeError, match="not writable"):
        util.fetch_dataset("toy", url_base="http://127.0.0.1:1/unused")


def test_fetch_dataset_warns_when_sha_missing(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    url_base, _archive, _sha = fake_release
    # patched_manifest already has sha256="", so the warning branch triggers.
    util.fetch_dataset("toy", url_base=url_base)
    err = capsys.readouterr().err
    assert "no SHA-256 recorded" in err


def test_fetch_dataset_force_redownloads(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
) -> None:
    url_base, _archive, sha = fake_release
    util.DATASETS["toy"] = util.DatasetSpec(**{**patched_manifest.__dict__, "sha256": sha})

    first = util.fetch_dataset("toy", url_base=url_base)
    (first / "hello.txt").write_text("mutated")

    second = util.fetch_dataset("toy", url_base=url_base, force=True)
    assert second == first
    # The force path wiped the cached directory and re-extracted the archive.
    assert (second / "hello.txt").read_bytes() == b"hi\n"


def test_fetch_dataset_reports_missing_toplevel_dir(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url_base, _archive, sha = fake_release
    util.DATASETS["toy"] = util.DatasetSpec(**{**patched_manifest.__dict__, "sha256": sha})

    # Stub the extractor so the archive never actually unpacks a ``toy/`` dir;
    # fetch_dataset must then raise a helpful RuntimeError.
    monkeypatch.setattr(util, "_safe_extract_tar", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="did not extract"):
        util.fetch_dataset("toy", url_base=url_base)


def test_fetch_datasets_returns_paths_in_order(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url_base, _archive, sha = fake_release
    util.DATASETS["toy"] = util.DatasetSpec(**{**patched_manifest.__dict__, "sha256": sha})
    monkeypatch.setenv("FIM_DATA_URL_BASE", url_base)

    paths = util.fetch_datasets(["toy"])
    assert paths == [isolated_cache / "toy"]


# ---------------------------------------------------------------------------
# CLI: ``python -m fim.util {list|where|fetch}``
# ---------------------------------------------------------------------------
def test_cli_list_prints_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    rc = util.main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    # Every built-in dataset should appear in the listing.
    for name in ("80um", "HGO", "NH", "exp2-benchmark"):
        assert name in out


def test_cli_where_prints_cache(
    isolated_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = util.main(["where"])
    assert rc == 0
    assert str(isolated_cache.resolve()) in capsys.readouterr().out


def test_cli_fetch_downloads(
    isolated_cache: Path,
    patched_manifest: util.DatasetSpec,
    fake_release: tuple[str, bytes, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    url_base, _archive, sha = fake_release
    util.DATASETS["toy"] = util.DatasetSpec(**{**patched_manifest.__dict__, "sha256": sha})

    rc = util.main(["fetch", "toy", "--url-base", url_base])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(isolated_cache / "toy") in out
    assert (isolated_cache / "toy" / "hello.txt").exists()
