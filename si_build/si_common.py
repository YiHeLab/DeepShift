#!/usr/bin/env python3
"""Shared plumbing for the supplementary tables.

Nothing is typed into a table: each is computed from artifacts on disk. Every root
below is an environment variable, so pointing the tables at another run is
`SI_COMPETITOR_ROOT=... python si_*.py`, not an edit.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

# The release checkout these generators sit in. They are run from this directory
# (`python <script>.py`), so the package and the repository root go on the path here,
# without asking the caller to set PYTHONPATH; the root is for `benchmarks`, which
# Table S11 and emit_text_numbers.py import.
REPO = Path(__file__).resolve().parent.parent
for directory in (REPO / "src", REPO):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deepshift.quality_control import P100Policy, load_p100_cutoffs


# --------------------------------------------------------------------------
# Roots. Change by environment, never by editing a script.
# --------------------------------------------------------------------------
def root(var: str) -> Path:
    """A root supplied by the environment, with no default.

    An unset variable resolves to a visibly invalid sentinel, never to a path on the
    machine that built this: the module still imports and ``--help`` still works, and the
    first attempted read fails naming the variable it wanted. Every generator in this
    directory takes its roots from here so there is one place to point at a build.
    """
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else Path(f"__{var}_REQUIRED__")


BUILD_ROOT = root("DEEPSHIFT_RUN")
COMPETITOR_ROOT = root("SI_COMPETITOR_ROOT")
ARCHIVE_ROOT = root("DEEPSHIFT_ARCHIVE")

DATA = BUILD_ROOT / "data"
CACHE = BUILD_ROOT / "work" / "unified_cache"


def archive(*parts: str) -> Path:
    """A finished measurement under the archive root."""
    return ARCHIVE_ROOT.joinpath(*parts)



def p100_cutoffs_path() -> Path:
    """The fitted p100 cutoff file of the build these tables describe, named by
    SI_P100_CUTOFFS. There is no packaged table and no default."""
    value = os.environ.get("SI_P100_CUTOFFS")
    if not value:
        raise OSError(
            "SI_P100_CUTOFFS must name the build's fitted p100 cutoff file")
    return Path(value).expanduser().resolve()


def p100_policy() -> P100Policy:
    """The p100 range rule of the build, read through the package's one implementation."""
    return P100Policy(load_p100_cutoffs(p100_cutoffs_path()))

# Manuscript-side inputs a generator checks itself against: the table it rewrites and the
# caption it copies through. They are small and belong to this release.
REFERENCE_INPUTS = Path(__file__).resolve().parent / "reference_inputs"


def reference_input(var: str, name: str = "") -> Path:
    """The vendored copy of a manuscript-side input, or ``$var`` when one is set.

    Unlike :func:`root`, this one has a default, because the file is in the repository.
    ``$var`` still points a regeneration at a newer deployed file without an edit.
    """
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else REFERENCE_INPUTS / name

# Where the .tex lands. Overridable so a regeneration against a different
# competitor root can be written elsewhere without overwriting the deployed source.
OUT = Path(os.environ.get("SI_OUT", str(Path(__file__).resolve().parent / "out")))

# --------------------------------------------------------------------------
# LaTeX helpers
# --------------------------------------------------------------------------
NUC_TEX = {
    "HA": r"H$\alpha$", "H": r"H$^{\mathrm{N}}$", "CA": r"C$\alpha$",
    "CB": r"C$\beta$", "C": r"C$'$", "N": "N",
}


def tex_nucleus(name: str) -> str:
    """Render an atom name (HA, CB, HD21, ne_shift, ...) for a table column."""
    key = name.replace("_shift", "").upper()
    if key in NUC_TEX:
        return NUC_TEX[key]
    element, rest = key[0], key[1:]
    if not rest:
        return element
    greek = {"A": r"\alpha", "B": r"\beta", "G": r"\gamma", "D": r"\delta",
             "E": r"\epsilon", "Z": r"\zeta", "H": r"\eta"}
    if rest[0] in greek:
        sub = greek[rest[0]] + (rest[1:] if len(rest) > 1 else "")
        return f"{element}$_{{{sub}}}$"
    return key


def num(x, digits=3, dash="---"):
    if x is None:
        return dash
    try:
        if x != x:                      # NaN
            return dash
    except TypeError:
        return dash
    return f"{x:.{digits}f}"


def thousands(n) -> str:
    return f"{int(n):,}"


def tex_thousands(n) -> str:
    """Grouped for LaTeX. A bare comma picks up math-mode spacing; ``{,}`` does not."""
    return f"{int(n):,}".replace(",", "{,}")


#: The authors' captions, one file per table (the text inside ``\caption{...}``). No
#: caption in this supplement was written by a generative AI tool, so no generator writes
#: one; the authors' text lives here and is spliced over the generator's, so the deployed
#: table is the generator's output byte for byte rather than a copy edited by hand.
CAPTIONS = Path(os.environ.get("SI_CAPTIONS_DIR")
                or Path(__file__).resolve().parent / "reference_inputs" / "captions")


def caption_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the one ``\\caption{...}`` group, braces balanced."""
    k = text.find("\\caption{")
    if k < 0:
        return None
    if text.find("\\caption{", k + 1) >= 0:
        raise ValueError("more than one \\caption in one table")
    j, depth = k + len("\\caption"), 0
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k, j + 1
        j += 1
    raise ValueError("unbalanced \\caption")


def splice_caption(tex: str, caption: str) -> str:
    """``tex`` with its caption replaced by ``caption``."""
    span = caption_span(tex)
    if span is None:
        raise ValueError("the table has no \\caption to replace")
    return tex[:span[0]] + "\\caption{" + caption + "}" + tex[span[1]:]


def portable_comments(text: str) -> str:
    """Comment lines name a root by its variable, never by a path on this machine."""
    return "\n".join(portable(line) if line.lstrip().startswith("%") else line
                     for line in text.split("\n"))


def portable(text: str) -> str:
    """``text`` with every root named by its variable, never by a path on this machine,
    and a temporary directory as ``$TMPDIR``."""
    end = r"(?=[/\s\"',;:)\]}]|$)"
    roots = []
    for var, path in (("$DEEPSHIFT_ARCHIVE", ARCHIVE_ROOT), ("$DEEPSHIFT_RUN", BUILD_ROOT),
                      ("$DEEPSHIFT_REPO", Path(__file__).resolve().parent.parent)):
        for spelling in {str(path), str(path).rstrip("/")}:
            if not spelling.startswith("__"):
                roots.append((re.compile(re.escape(spelling) + end), var))
    for var, env in (("$DEEPSHIFT_ARCHIVE", "DEEPSHIFT_ARCHIVE"),
                     ("$DEEPSHIFT_RUN", "DEEPSHIFT_RUN")):
        if os.environ.get(env):
            roots.append((re.compile(re.escape(os.environ[env].rstrip("/")) + end), var))
    roots.sort(key=lambda r: -len(r[0].pattern))      # the deepest root first
    roots.append((re.compile(re.escape(tempfile.gettempdir()) + r"/tmp[A-Za-z0-9_]{6,}"
                             + end), "$TMPDIR"))
    roots.append((re.compile(re.escape(str(Path.home())) + end), "~"))
    for pattern, var in roots:
        text = pattern.sub(lambda _m, v=var: v, text)
    return text


def emit(stem: str, tex: str, *, script: str, banner: str | None = None,
         placeholder_caption: bool = True) -> Path:
    """Write out/<stem>.tex.

    ``placeholder_caption=False`` for a table that carries a real caption -- the
    author's, or a short title the banner explains.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / f"{stem}.tex"
    caption_file = CAPTIONS / f"{stem}.tex"
    authors = caption_file.is_file()
    if authors:
        tex = splice_caption(tex, caption_file.read_text().strip("\n"))
    header = [f"% GENERATED by si_build/{script} -- do not hand-edit; re-run the script."]
    header += ([
        "% No caption in this supplement was written by a generative AI tool. The",
        f"% caption is the authors', from si_build/reference_inputs/captions/{stem}.tex.",
    ] if authors else [
        "% Captions are placeholders: no caption in this supplement was",
        "% written by a generative AI tool. The facts each caption needs sit in the",
        "% %-comment beneath its table.",
    ] if placeholder_caption else
        ["% No caption in this supplement was written by a generative AI tool."])
    if banner is not None:
        header.append(banner.rstrip("\n"))
    tex_path.write_text(portable_comments(
        "\n".join(header) + "\n\n" + tex.rstrip() + "\n"))
    print(f"  wrote {tex_path}")
    return tex_path


PLACEHOLDER_CAPTION = r"\note{Caption is yours --- facts in the comment below.}"


def table(caption_note: str, label: str, body: str, *, small=True,
          resize=False, position="H", longtable=None, head="", foot="",
          caption=None, size=None) -> str:
    """One float. The caption is a placeholder.

    ``longtable`` is a column specification: the table then runs across pages, ``head``
    repeats at the top of each, ``foot`` is a full \\endfoot/\\endlastfoot block, and
    ``size`` (e.g. ``\\footnotesize``) wraps the whole thing in a group.
    """
    cap = PLACEHOLDER_CAPTION if caption is None else caption
    if longtable:
        parts = [f"{{{size}\n" if size else "",
                 f"\\begin{{longtable}}{{{longtable}}}\n",
                 f"\\caption{{{cap}}}\\label{{{label}}}\\\\\n"]
        if head:
            for terminator in ("\\endfirsthead\n", "\\endhead\n"):
                parts += ["\\toprule\n", head.rstrip() + "\n", "\\midrule\n", terminator]
        if foot:
            parts.append(foot.rstrip() + "\n")
        parts += [body.rstrip() + "\n", "\\end{longtable}\n", "}\n" if size else ""]
        return "".join(parts) + "\n" + caption_note.rstrip() + "\n\n"
    float_size = r"\small" if small else ""
    open_ = f"\\begin{{table}}[{position}]\\centering{float_size}\n"
    cap_block = f"\\caption{{{cap}}}\n\\label{{{label}}}\n"
    inner = body
    if resize:
        inner = "\\resizebox{\\textwidth}{!}{%\n" + body + "}\n"
    return open_ + cap_block + inner + "\\end{table}\n\n" + caption_note.rstrip() + "\n\n"
