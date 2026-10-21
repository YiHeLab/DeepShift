"""The confidence figure builds its cohort with the package's matched-cell rule."""
import ast
from pathlib import Path

FIGURE = Path(__file__).resolve().parents[1] / "figures" / "fig_plddt_source.py"


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name):
                yield f"{value.id}.{node.func.attr}"


def test_the_confidence_figure_takes_its_matched_cohort_from_the_package():
    """The matched cell set is one rule, defined in deepshift.cohorts, so the figure and
    the source-comparison figure beside it cannot describe different cohorts."""
    tree = ast.parse(FIGURE.read_text())
    calls = set(_calls(tree))
    assert "cohorts.matched_cells" in calls
    assert not any(call.endswith(".matched_cells") and call != "cohorts.matched_cells"
                   for call in calls)
