"""DeepShift's lead over the runner-up in every Table 1 cell, at 5 % and after Bonferroni.

One function for the two places that count these leads: ``figures/fig_backbone_benchmark.py``
(Figure 2 / Table 1's record) and ``si_build/emit_text_numbers.py`` (main.tex's "16 of 18 ...
17 of 18 at alpha = 0.05" and the add-one count beside them), so the printed counts and the
figure's record cannot come from two implementations.
"""
from __future__ import annotations

#: The backbone nuclei and structure sources of Table 1, in its order.
NUCLEI = ("CA", "CB", "C", "HA", "H", "N")
SOURCES = ("xtal", "af", "nmr")
#: The family-wise level of main.tex's "significant after Bonferroni correction".
ALPHA = 0.05
#: Table 1's printed rows: the arms fig_backbone_benchmark draws and tabulates, and MLFF,
#: typed into main.tex from Table S12's record of the same marginal report.
TABLE1_ARMS = ("deepshift", "ucbshift_full_sameprot", "ucbshift_x", "shiftx2_sameprot",
               "sparta", "mlff")


def lead_tests(marginal: dict, paired: dict, competitors) -> dict:
    """DeepShift's lead over the runner-up in every source x backbone nucleus, tested at
    the 5 % level and Bonferroni-corrected over those cells.

    The runner-up is the lowest-RMSE arm of ``competitors``. Each test reads the paired
    report's bootstrap: ``k`` of its replicates do not favour DeepShift. A lead is
    significant at 5 % when the 95 % paired interval excludes zero, and after Bonferroni
    when the one-sided tail k/B falls below alpha / (2 x tests), i.e. the two-sided
    1 - alpha/tests interval excludes zero; (k + 1)/(B + 1) in place of k/B is recorded
    beside it.
    """
    boot = paired["boot"]
    cells = [(src, nuc) for src in SOURCES for nuc in NUCLEI]
    threshold = ALPHA / (2 * len(cells))
    tests = []
    for src, nuc in cells:
        block = marginal["by_source"][src]["nuclei"][nuc]
        runner = min(competitors, key=lambda arm: block[arm]["rmse"])
        node = paired[src][nuc][runner]
        k = (1 - node["p_deepshift_better"]) * boot
        if abs(k - round(k)) > 1e-6:
            raise SystemExit(f"{src}/{nuc}: P(DeepShift better) is not a count out of "
                             f"{boot} replicates, so k cannot be read from it")
        k = round(k)
        leads = node["diff"] < 0          # on the cells both arms of the pair emitted
        tests.append({
            "source": src, "nucleus": nuc, "runner_up": runner, "deepshift_leads": leads,
            "diff": node["diff"], "ci95": node["ci"], "k_not_better": k,
            "significant_alpha": leads and node["ci"][1] < 0,
            "significant_bonferroni": leads and k / boot < threshold,
            "significant_bonferroni_plus_one": leads and (k + 1) / (boot + 1) < threshold})
    return {
        "runner_up_among": list(competitors),
        "n_tests": len(cells), "alpha": ALPHA, "bonferroni_alpha": ALPHA / len(cells),
        "boot": boot, "unit": paired["unit"],
        "n_deepshift_leads": sum(t["deepshift_leads"] for t in tests),
        "n_significant_alpha": sum(t["significant_alpha"] for t in tests),
        "n_significant_bonferroni": sum(t["significant_bonferroni"] for t in tests),
        "n_significant_bonferroni_plus_one": sum(t["significant_bonferroni_plus_one"]
                                                 for t in tests),
        "tests": tests,
    }


def table1_competitors(paired: dict) -> list:
    """Table 1's printed rows other than the reference arm."""
    return [arm for arm in TABLE1_ARMS if arm != paired["reference_arm"]]
