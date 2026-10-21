"""Every tracked file names its roots by variable, not by a path on one machine."""

import importlib
import shutil
import subprocess
from pathlib import Path

import pytest

from deepshift.data import paths

REPO = Path(__file__).resolve().parents[1]
BINARY_SUFFIXES = {".pt", ".parquet", ".gz", ".pdb", ".ipynb", ".png", ".pdf"}
# A path rooted in a user account, or in the scratch directory of one machine, is a path
# on one machine; a producer takes its roots from the run variable or from an argument.
# The names are joined and never written out so that this sweep does not report itself.
HOME_ROOTS = tuple(f"/{name}/" for name in ("home", "Users", "users", "root", "tmp"))
# One variable names the run every producer reads, and no producer, document or sidecar
# names it by a second name: two names for one run are two lineages to keep aligned.
RUN_VARIABLE = "DEEPSHIFT_RUN"
OTHER_RUN_NAMES = ("SHIFT_RUN_ROOT", "SI_BUILD_ROOT", "DEEPSHIFT_RESCORE_ROOT",
                   "DEEPSHIFT_BENCH_ROOT", "DEEPSHIFT_MANUSCRIPT_BUILD")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def tracked_text_files():
    """The tracked files; an unpacked release archive has no tracking, so the sweep skips."""
    if not shutil.which("git") or git("rev-parse", "--show-toplevel").stdout.strip() != str(REPO):
        pytest.skip("not a git checkout of this repository")
    paths = [REPO / name for name in git("ls-files").stdout.split()]
    return [path for path in paths
            if path.is_file() and path.suffix not in BINARY_SUFFIXES]


def read(path):
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return ""


def test_no_tracked_file_names_a_path_outside_the_repository():
    offenders = []
    for path in tracked_text_files():
        for number, line in enumerate(read(path).splitlines(), start=1):
            if any(root in line for root in HOME_ROOTS):
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert offenders == []


def test_no_tracked_file_names_the_run_by_a_second_variable():
    offenders = []
    for path in tracked_text_files():
        for number, line in enumerate(read(path).splitlines(), start=1):
            for name in OTHER_RUN_NAMES:
                if name in line and path.name != Path(__file__).name:
                    offenders.append(f"{path.relative_to(REPO)}:{number} names {name}")
    assert offenders == []


def test_the_run_root_resolves_from_the_one_variable(monkeypatch, tmp_path):
    """``deepshift.data.paths`` is where the run is resolved, and the figures and the
    supplementary generators resolve it through that module's variable."""
    monkeypatch.setenv(RUN_VARIABLE, str(tmp_path))
    try:
        assert str(tmp_path) == importlib.reload(paths).RUN_ROOT
    finally:
        monkeypatch.undo()
        importlib.reload(paths)
