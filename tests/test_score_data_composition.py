"""Table S22's two records, and the contract the emitter reads them through.

The table's producer and its emitter are separate programs joined only by the shape of
two JSON files, which is exactly the join that breaks without raising: a renamed key
gives a KeyError in the SI build hours later, and a cohort applied as a row filter
instead of a cell mask silently repairs the pairing and reports a comparison over
different residues. The tests below run the scorer on synthetic held-out dumps whose
right answer is known by construction, then read its output with the emitter's own
loader.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks import score_data_composition as sdc
from deepshift import evaluate
from deepshift.domain import VALID_SOURCES
from deepshift.normalization import BACKBONE_SHIFTS

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"
COLUMNS = list(BACKBONE_SHIFTS)
ROWS_PER_STRUCTURE = 20
#: Label sets every source realizes, then one realized by AlphaFold alone. The second
#: group is what the matched cohort has to drop.
MATCHED_LABEL_SETS = 8
AF_ONLY_LABEL_SETS = 3


def _corpus(seed=0):
    """One held-out fold: rows tagged by label set and source, with true ppm values."""
    rng = np.random.default_rng(seed)
    label_set, source, keys = [], [], []
    for index in range(MATCHED_LABEL_SETS):
        for code, _name in enumerate(VALID_SOURCES):
            for row in range(ROWS_PER_STRUCTURE):
                label_set.append(f"ls{index}")
                source.append(code)
                keys.append(f"ls{index}|{code}|{row}")
    for index in range(AF_ONLY_LABEL_SETS):
        for row in range(ROWS_PER_STRUCTURE):
            label_set.append(f"af{index}")
            source.append(0)
            keys.append(f"af{index}|0|{row}")
    n = len(keys)
    true = rng.normal(0.0, 2.0, (n, len(COLUMNS)))
    return (np.asarray(label_set), np.asarray(source), np.asarray(keys),
            np.asarray([int(x) for x in rng.integers(0, 4, n)]), true)


def _dump(path, corpus, error_sd, seed):
    """One arm's trainer-shaped dump: the same rows, its own prediction error."""
    label_set, source, keys, residue, true = corpus
    rng = np.random.default_rng(seed)
    pred = true + rng.normal(0.0, error_sd, true.shape)
    np.savez(path, pred_ppm=pred, true_ppm=true, resolved_true_ppm=true,
             mask=np.ones_like(true, dtype=bool),
             effective_mask=np.ones_like(true, dtype=bool),
             ambiguity=np.zeros_like(true, dtype=np.int64), residue_code=residue,
             shift_cols=np.asarray(COLUMNS), label_residue_id=keys,
             materialization_id=np.asarray([k.split("|")[0] + k.split("|")[1] for k in keys]),
             label_set_id=label_set, bmrb=label_set, source=source)
    return path


def _reports(directory, arms_report=True):
    """The construction reports the emitter reads beside the scorer's own records."""
    directory.mkdir(parents=True, exist_ok=True)
    if arms_report:
        (directory / "arms_report.json").write_text(
            '{"heldout_fold": 1, "train_folds": [2, 3, 4, 5], "arms": {'
            '"floor": {"n_materializations": 911, "n_label_sets": 911, "n_rows": 77408},'
            '"redundant": {"n_materializations": 2733, "n_label_sets": 911, "n_rows": 231902},'
            '"diverse": {"n_materializations": 2544, "n_label_sets": 2544, "n_rows": 231947}}}')
    else:
        (directory / "one_per_protein_report.json").write_text(
            '{"heldout_fold": 1, "train_folds": [2, 3, 4, 5],'
            ' "rule": "one structure per label set, uniform random",'
            ' "pool": {"label_sets": 8022, "structures": 12976, "rows": 1144786},'
            ' "arm": {"structures": 8022, "rows": 714322}}')


def _run(tmp_path):
    """Score five synthetic arms whose accuracy order is set by construction."""
    corpus = _corpus()
    div, rep = tmp_path / "data_diversity", tmp_path / "source_transfer"
    _reports(div, arms_report=True)
    _reports(rep, arms_report=False)
    # Error falls from floor to redundant to diverse, and the paper member is the best
    # arm of all, so every contrast below has a known sign.
    paths = {name: _dump(tmp_path / f"{name}.npz", corpus, sd, seed)
             for seed, (name, sd) in enumerate(
                 (("floor", 1.6), ("redundant", 1.3), ("diverse", 1.0),
                  ("one_per_protein", 1.2), ("shipped_all", 0.9)))}
    sdc.main([
        "--floor", str(paths["floor"]), "--redundant", str(paths["redundant"]),
        "--diverse", str(paths["diverse"]),
        "--one-per-protein", str(paths["one_per_protein"]),
        "--shipped-all", str(paths["shipped_all"]),
        "--arms-report", str(div / "arms_report.json"),
        "--out-comparison", str(div / "arms_comparison.json"),
        "--out-source-matched", str(rep / "source_matched_report.json"),
        "--boots", "32"])
    return div, rep


def _emitter(monkeypatch, div, rep, out_dir):
    monkeypatch.setenv("SI_DATA_DIVERSITY_DIR", str(div))
    monkeypatch.setenv("SI_SOURCE_TRANSFER_DIR", str(rep))
    monkeypatch.setenv("SI_OUT", str(out_dir))
    sys.path.insert(0, str(SI_BUILD))
    try:
        for name in ("si_common", "emit_data_composition"):
            sys.modules.pop(name, None)
            spec = importlib.util.spec_from_file_location(name, SI_BUILD / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules["emit_data_composition"]
    finally:
        sys.path.remove(str(SI_BUILD))


def test_the_emitter_reads_both_records_the_scorer_writes(monkeypatch, tmp_path):
    """CONTRACT: Table S22's loader consumes the scorer's output unchanged.

    Every key the table prints is read here from the files the scorer just wrote, so a
    renamed contrast or a dropped per-nucleus field fails now rather than in the SI build.
    """
    div, rep = _run(tmp_path)
    module = _emitter(monkeypatch, div, rep, tmp_path / "out")
    block1, block2 = module.load()

    assert block1["heldout_fold"] == 1 and block1["train_folds"] == [2, 3, 4, 5]
    assert block1["rows"] == {"floor": 77408, "redundant": 231902, "diverse": 231947}
    # The floor's RMSE reaches the table through two different contrasts, and the
    # emitter refuses them if they disagree; that check is what pins the record shape.
    for column in COLUMNS:
        assert block1["floor"][column] > block1["redundant"][column] > block1["diverse"][column]
        assert block1["cells"][column] > 0
    for source in VALID_SOURCES:
        per_source = block2["per_source"][source]
        assert set(per_source["one_per_protein"]) == set(COLUMNS)
        # shipped_all was built with the smaller error, so keeping every structure
        # gives the lower RMSE.
        assert all(per_source["every_structure"][column] < per_source["one_per_protein"][column]
                   for column in COLUMNS)


def test_the_matched_cohort_drops_a_label_set_only_one_source_realizes(tmp_path):
    """INVARIANT: block 2's cohort is the label sets all three sources realize.

    Scoring the marginal AlphaFold rows instead would compare the three sources over
    different proteins, which is the confound the matched cohort exists to remove.
    """
    _, rep = _run(tmp_path)
    report = json.loads((rep / "source_matched_report.json").read_text())
    af = report["cohorts"]["af"]
    assert af["marginal_label_sets"] == MATCHED_LABEL_SETS + AF_ONLY_LABEL_SETS
    assert af["matched_label_sets"] == MATCHED_LABEL_SETS
    assert af["matched_rows"] == MATCHED_LABEL_SETS * ROWS_PER_STRUCTURE
    for column in COLUMNS:
        matched = report["table_a"]["matched"]["af"]["shipped_all"][column]
        marginal = report["table_a"]["marginal"]["af"]["shipped_all"][column]
        assert matched["n"] < marginal["n"]
        assert matched["n_label_sets"] == MATCHED_LABEL_SETS


def test_a_cohort_is_a_cell_mask_so_the_paired_comparison_keeps_its_rows(tmp_path):
    """INVARIANT: restricting to a source must not renumber the rows.

    ``compare`` refuses two arms whose rows do not line up, so a cohort applied as a row
    filter would either raise or, worse, pair each arm's Nth surviving row with the
    other's -- a comparison over different residues that looks entirely normal.
    """
    corpus = _corpus()
    arms = {name: evaluate.read_predictions(_dump(tmp_path / f"{name}.npz", corpus, sd, seed))
            for seed, (name, sd) in enumerate((("a", 1.0), ("b", 0.8)))}
    rows = arms["a"].source == VALID_SOURCES.index("xtal")
    restricted = {name: sdc._restrict(preds, rows) for name, preds in arms.items()}
    assert all(preds.n_rows == arms["a"].n_rows for preds in restricted.values())
    result = evaluate.compare(restricted["a"], restricted["b"], columns=COLUMNS, boots=32)
    assert set(result) == set(COLUMNS)
    assert all(entry["n"] == int(rows.sum()) for entry in result.values())


def test_arms_are_aligned_on_their_shared_rows(tmp_path):
    """INVARIANT: an arm carrying a row the others lack is aligned away, not mispaired."""
    corpus = _corpus()
    label_set, source, keys, residue, true = corpus
    wider = (np.append(label_set, "extra"), np.append(source, 0),
             np.append(keys, "extra|0|0"), np.append(residue, 0),
             np.vstack([true, true[:1]]))
    narrow = evaluate.read_predictions(_dump(tmp_path / "narrow.npz", corpus, 1.0, 0))
    wide = evaluate.read_predictions(_dump(tmp_path / "wide.npz", wider, 1.0, 1))
    aligned = sdc.common_rows({"narrow": narrow, "wide": wide})
    assert aligned["narrow"].n_rows == aligned["wide"].n_rows == narrow.n_rows
    np.testing.assert_array_equal(aligned["narrow"].row_key, aligned["wide"].row_key)
