"""Every command the protocol issues parses against the real command surface."""
import importlib.util
import re
import shlex
import sys
from pathlib import Path

import pytest
from test_fit_p100_oof64 import FITTER

from deepshift import quality_control
from deepshift.pipeline import build_parser, main

REPO = Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "PROTOCOL.md"
_SCRIPTS: dict[str, object] = {}


def script(name):
    """The release script of that name, loaded once for its argument parser."""
    if name not in _SCRIPTS:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SCRIPTS[name] = module
    return _SCRIPTS[name]


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        main([])


@pytest.mark.parametrize("argv", [
    ["predict", "--help"], ["train", "--help"], ["evaluate", "--help"],
    ["data", "build", "--help"],
    ["intervals", "fit", "--help"], ["intervals", "evaluate", "--help"],
    ["protocol", "--help"],
])
def test_every_subcommand_is_wired(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


def _expand_braces(token):
    """``name{1,2,3}.npz`` -> the three tokens the shell would produce."""
    match = re.search(r"\{([^{}]*,[^{}]*)\}", token)
    if not match:
        return [token]
    head, tail = token[:match.start()], token[match.end():]
    return [head + item + tail for item in match.group(1).split(",")]


def _documented_commands():
    """Every command the protocol issues, as the shell would hand it to the program:
    loop variables and environment paths filled in, brace lists expanded."""
    commands, block, current = [], False, []
    for line in DOCUMENT.read_text().splitlines():
        if line.startswith("```"):
            block = not block
            continue
        if not block or not line.strip() or line.lstrip().startswith(
                ("#", "export ", "for ", "done", "cmp ", "cp ", "mkdir ")):
            continue
        current.append(line.rstrip("\\").strip())
        if line.rstrip().endswith("\\"):
            continue
        text = " ".join(current)
        current = []
        text = text.replace("${fold}", "1").replace("$fold", "1")
        text = re.sub(r"\$\{?[A-Z_]+\}?", "/run", text)
        argv = [expanded for token in shlex.split(text) for expanded in _expand_braces(token)]
        if argv[:2] == ["python", "-c"]:
            continue                      # an operator's own shell one-liner, not a program here
        commands.append(argv)
    return commands


@pytest.mark.parametrize("argv", _documented_commands(), ids=lambda argv: " ".join(argv[:3]))
def test_every_documented_command_parses(argv, monkeypatch):
    if argv[0] == "deepshift":
        build_parser().parse_args(argv[1:])
    elif argv[:3] == ["python", "-m", "deepshift.quality_control"]:
        quality_control.build_parser().parse_args(argv[3:])
    elif argv[0] == "python" and argv[1].endswith("fit_p100_oof64.py"):
        monkeypatch.setattr(sys, "argv", argv[1:])
        FITTER.parse_args()
    elif argv[0] == "python" and argv[1].endswith("export_dataset.py"):
        script("export_dataset").parse_args(argv[2:])
    elif argv[0] == "pip":
        assert argv[1:3] == ["install", "-e"]
    else:
        raise AssertionError(f"the protocol issues a command this test does not know: {argv}")


def test_the_protocol_document_names_every_deepshift_subcommand_it_needs():
    heads = {tuple(argv[1:3]) for argv in _documented_commands() if argv[0] == "deepshift"}
    assert {("data", "build"), ("train", "--cache"),
            ("evaluate", "--cache"), ("intervals", "fit"), ("intervals", "evaluate"),
            ("protocol", "--out")} <= heads
