"""Resolve the two non-Python tools used by the data build."""
from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import subprocess
import tarfile
import urllib.request
from pathlib import Path

MMSEQS_COMMIT = "4fe20e6bb0aa219a35f6190322a91ab19cb6898d"
MMSEQS_URL = f"https://mmseqs.com/archive/{MMSEQS_COMMIT}/mmseqs-linux-avx2.tar.gz"
MMSEQS_ARCHIVE_SHA256 = "74f1bc7fddadff503e0a589d9d65085c3b6da45b0b31de58ab3b7ca2c5154a98"
MMSEQS_BINARY_SHA256 = "6220c0adec5bb994ed480cbe0689c42fd8481689ae2b5b3404143bb5d68e8fe8"
SHIFTX_URL = "https://shiftx.wishartlab.com/download/shiftx.tar.gz"
SHIFTX_ARCHIVE_SHA256 = "a08bc301b119154675eb05d9c5c69b7f582e8c2fa0cc81c5040a6ac013ac1965"


def _sha256(path: Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}: {path}")
    return path


def ensure_mmseqs(tool_dir: str | Path) -> Path:
    """Return MMseqs2, downloading the reference static build when absent."""
    configured = os.environ.get("MMSEQS_BIN")
    if configured:
        return _require_hash(Path(configured), MMSEQS_BINARY_SHA256, "MMseqs2 executable")
    root = Path(tool_dir)
    binary = root / "mmseqs" / "bin" / "mmseqs"
    if binary.is_file():
        return _require_hash(binary, MMSEQS_BINARY_SHA256, "MMseqs2 executable")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "mmseqs-linux-avx2.tar.gz"
    print(f"fetch {MMSEQS_URL} -> {archive}")
    urllib.request.urlretrieve(MMSEQS_URL, archive)
    actual = _sha256(archive)
    if actual != MMSEQS_ARCHIVE_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"MMseqs2 archive hash mismatch: {actual}")
    with tarfile.open(archive, "r:gz") as bundle:
        destination = root.resolve()
        for member in bundle.getmembers():
            if not (destination / member.name).resolve().is_relative_to(destination):
                raise RuntimeError(f"unsafe MMseqs2 archive member: {member.name}")
        bundle.extractall(root)
    return _require_hash(binary, MMSEQS_BINARY_SHA256, "MMseqs2 executable")


def ensure_shiftx(tool_dir: str | Path | None = None) -> Path:
    """Download, build and return the CPU-only SHIFTX 1.1 executable.

    The source tree is fetched from the authors' server and compiled here; it is not
    redistributed. SHIFTX_SOURCE points at an existing source tree; SHIFTX_BIN at a
    built executable.
    """
    configured = os.environ.get("SHIFTX_BIN")
    if configured:
        return Path(configured)
    parent = Path(tool_dir or os.environ.get(
        "SHIFT_TOOL_DIR", os.path.join(os.getcwd(), ".deepshift-tools")))
    root = parent / "shiftx"
    binary = root / "shiftx"
    if binary.is_file():
        return binary
    source = os.environ.get("SHIFTX_SOURCE")
    if source:
        if not root.is_dir():
            shutil.copytree(source, root)
    elif not root.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        archive = parent / "shiftx.tar.gz"
        if not archive.is_file() or _sha256(archive) != SHIFTX_ARCHIVE_SHA256:
            print(f"fetch {SHIFTX_URL} -> {archive}")
            # The server's certificate does not validate; the pinned SHA-256 authenticates the archive.
            context = ssl._create_unverified_context()
            with (urllib.request.urlopen(SHIFTX_URL, context=context, timeout=120) as response,
                  archive.open("wb") as output):
                shutil.copyfileobj(response, output)
        _require_hash(archive, SHIFTX_ARCHIVE_SHA256, "SHIFTX archive")
        with tarfile.open(archive, "r:gz") as bundle:
            destination = parent.resolve()
            for member in bundle.getmembers():
                if not (destination / member.name).resolve().is_relative_to(destination):
                    raise RuntimeError(f"unsafe SHIFTX archive member: {member.name}")
            bundle.extractall(parent)
    subprocess.run(["make", "-C", str(root)], check=True)
    if not binary.is_file():
        raise RuntimeError(f"SHIFTX build produced no executable at {binary}")
    return binary
