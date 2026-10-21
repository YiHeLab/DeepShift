"""The text-numbers record recomputes each number from the run's files.

The shift-list numbers re-derive the fetch stage's disagreement flag and refuse a file
whose flag they do not reproduce; the rereferencing numbers come from the per-source fits
of scripts/reref_diagnostics.py.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "si_build"))

SELECTION_HEADER = ["logical_entity_id", "label_set_id", "bmrb_id", "shift_list_id",
                    "selected", "reason", "n_conflict_cells",
                    "alternate_label_set_disagrees", "candidate_payload_json"]


def _payload(cells):
    return json.dumps({"observed_cells": [[pos, "ALA", atom, ppm] for pos, atom, ppm in cells]})


SELECTION = [
    ["le_1", "ls_a", "1", "1", "True", "selected", "0", "True",
     _payload([(1, "CA", 50.0), (2, "CA", 51.0)])],
    ["le_1", "ls_b", "1", "2", "False", "eligible_not_selected", "0", "True",
     _payload([(1, "CA", 50.1), (2, "CA", 51.0)])],
    ["le_2", "ls_c", "2", "1", "True", "selected", "3", "False",
     _payload([(1, "N", 120.0)])],
]
OFFSETS = {"ls_a": {"method": "computed", "offsets": {"CA": 0.5, "N": -0.2}},
           "ls_c": {"method": "computed", "offsets": {"CA": 0.3}},
           "ls_d": {"method": "computed", "offsets": {}}}
DIAGNOSTICS = {"per_label_set": {
    "ls_a": {"outcome": "offset", "struct_used": "af:A", "fits": {
        "af": {"CA": {"offset_ppm": 0.5, "residues": 10}, "N": {"offset_ppm": -0.2, "residues": 30}},
        "xtal": {"CA": {"offset_ppm": 0.4, "residues": 9}}}},
    "ls_c": {"outcome": "offset", "struct_used": "xtal:B", "fits": {
        "af": {"CA": {"offset_ppm": 0.1, "residues": 7}},
        "xtal": {"CA": {"offset_ppm": 0.3, "residues": 20}}}},
    "ls_d": {"outcome": "rounds_to_zero", "struct_used": "af:A", "fits": {
        "af": {"CA": {"offset_ppm": 0.01, "residues": 99}}}},
}}


@pytest.fixture
def producer(tmp_path, monkeypatch):
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))
    monkeypatch.delenv("SI_FINAL_SCORES_DIR", raising=False)   # a sourced release env must not leak in
    monkeypatch.setenv("DEEPSHIFT_RUN", str(tmp_path / "run"))
    for variable in ("SI_REREF_DIAGNOSTICS", "DEEPSHIFT_XEISD_PROTOCOL", "SI_SELF_RETRIEVAL_UCBY",
                     "SI_SELF_RETRIEVAL_SHIFTY61", "SI_GATE_CSPRED_TM", "DEEPSHIFT_REFDB_SHIFTS",
                     "DEEPSHIFT_REFDB_LIBRARY", "DEEPSHIFT_UCBSHIFT_LEAKAGE"):
        monkeypatch.delenv(variable, raising=False)
    data = tmp_path / "run" / "data"
    data.mkdir(parents=True)
    (data / "fetch_manifest.json").write_text(json.dumps(
        {"n_api_ids": 3, "n_logical_entities": 2}))
    with open(data / "label_set_selection.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SELECTION_HEADER)
        writer.writerows(SELECTION)
    (data / "reref_offsets.json").write_text(json.dumps(OFFSETS))
    (data / "reref_diagnostics.json").write_text(json.dumps(DIAGNOSTICS))
    for module in ("si_common", "emit_text_numbers"):
        sys.modules.pop(module, None)
    import emit_text_numbers as module
    return module, data


def test_shift_list_numbers_reproduce_the_disagreement_flag(producer):
    module, data = producer
    g = module.Group()
    module.shift_lists(g, data)
    v = {k: e["value"] for k, e in g.values.items()}
    assert v["depositions"] == 3
    assert v["depositions_without_usable_shifts"] == 1
    assert v["selected_label_sets"] == 2
    assert v["selected_with_a_disagreeing_alternate"] == 1
    assert (v["selected_lists_with_conflicting_cells"], v["conflicting_cells_in_selected_lists"]) == (1, 3)
    assert v["alternate_disagreement_cells"] == 1
    assert v["alternate_disagreement_median_ppm"] == pytest.approx(0.1)


def test_a_flag_the_payloads_do_not_reproduce_is_refused(producer):
    module, data = producer
    rows = [list(r) for r in SELECTION]
    rows[0][7] = "False"
    with open(data / "label_set_selection.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([SELECTION_HEADER, *rows])
    with pytest.raises(SystemExit, match="does not reproduce"):
        module.shift_lists(module.Group(), data)


def test_residues_per_offset_and_the_source_refit_come_from_the_fits(producer):
    module, data = producer
    g = module.Group()
    module.rereferencing(g, data)
    v = {k: e["value"] for k, e in g.values.items()}
    # ls_a: CA 10 and N 30 on af; ls_c: CA 20 on xtal; ls_d has no offset.
    assert v["offset_fits"] == 3
    assert v["offset_fit_median_residues"] == 20
    assert v["refit_af_vs_xtal_CA_label_sets"] == 2
    assert v["refit_af_vs_xtal_CA_median_abs_difference_ppm"] == pytest.approx(0.15)
    assert v["refit_af_vs_xtal_CA_pearson_r"] == pytest.approx(1.0)
    assert v["refit_af_vs_nmr_CA_pearson_r"] is None


def test_diagnostics_without_fits_are_refused(producer):
    module, data = producer
    bare = {"per_label_set": {k: {"outcome": r["outcome"], "struct_used": r["struct_used"]}
                              for k, r in DIAGNOSTICS["per_label_set"].items()}}
    (data / "reref_diagnostics.json").write_text(json.dumps(bare))
    with pytest.raises(SystemExit, match="no per-source fits"):
        module.rereferencing(module.Group(), data)


def test_dataset_shares_come_from_the_released_tables(producer, tmp_path):
    module, _data = producer
    pd = pytest.importorskip("pandas")
    released = tmp_path / "released"
    released.mkdir()
    pd.DataFrame({"logical_entity_id": ["e1", "e2", "e3", "e4"]}).to_parquet(
        released / "deepshift_sequences.parquet")
    pd.DataFrame({"label_set_id": ["a", "a", "b", "c"], "source": ["af", "xtal", "af", "nmr"]}
                 ).to_parquet(released / "deepshift_structures.parquet")
    pd.DataFrame({"label_set_id": ["a", "a", "b", "c", "c", "d"],
                  "bmrb_id": ["1", "1", "2", "3", "3", "4"],
                  "fold": pd.array([6, 6, 0, 1, 1, None], dtype="Int64")}
                 ).to_parquet(released / "deepshift_shifts.parquet")
    g = module.Group()
    module.dataset_shares(g, released)
    v = {k: e["value"] for k, e in g.values.items()}
    assert (v["bmrb_label_sets"], v["label_sets_realized"]) == (4, 3)
    assert v["label_sets_realized_pct"] == pytest.approx(75.0)
    assert (v["label_sets_with_experimental_structure"],
            v["label_sets_realized_by_alphafold_only"]) == (2, 1)
    # Fold 0 is outside the pool and a row with no fold is in neither.
    assert (v["holdout_label_sets"], v["holdout_pool_label_sets"]) == (1, 2)
    assert (v["holdout_residues"], v["holdout_pool_residues"]) == (2, 4)
    assert v["holdout_depositions_pct"] == pytest.approx(50.0)


def test_the_json_holds_every_value_and_what_was_not_recomputed(producer, tmp_path,
                                                                monkeypatch):
    module, _data = producer
    monkeypatch.setattr(module, "RELEASED", tmp_path / "no_release")
    module.main(["--allow-missing"])
    written = json.loads((tmp_path / "out" / "text_numbers.json").read_text())
    assert written["depositions"] == {"value": 3, "description": "n_api_ids"}
    assert written["offset_fit_median_residues"]["value"] == 20
    assert set(written["not_recomputed"]) == {
        "atom_cloud", "fold0_pin", "fence_effect", "table1_significance", "xeisd",
        "xeisd_pools", "intervals",
        "fold_balance", "unfiltered_n_extremes", "ucbshift_y_example",
        "source_matched", "three_source", "anticorrelated", "self_retrieval_ucbshift_y",
        "self_retrieval_shifty61", "environments", "dataset_shares"}


CLUSTER = {
    "members": {"4": {"slurm_job": "1_4", "node": "gpu01"}, "5": {"slurm_job": "1_5", "node": "gpu01"}},
    "node_gres": {"gpu01": "gpu:a100:2"},
    "environment": {"conda_env": "env271 (cluster)", "python": "3.11.15", "torch": "2.7.1+cu118",
                    "torch_cuda": "11.8", "numpy": "2.4.6"},
    "manifest_platform_check": "final/fold{4,5}.json manifest.platform = Linux-4.18.0-el8 (cluster)",
}


def _members(run, python="3.11.15 | conda"):
    (run / "final").mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        manifest = {"python": python if fold > 3 else "3.12.11 | conda",
                    "platform": "Linux-4.18.0-el8.x86_64" if fold > 3 else "Linux-6.8.0",
                    "command_argv": ["/opt/conda/envs/env271/bin/python" if fold > 3
                                     else "/venv/bin/python"]}
        (run / "final" / f"fold{fold}.json").write_text(json.dumps({"manifest": manifest}))
    (run / "final" / "members45_environment_20261007.json").write_text(json.dumps(CLUSTER))


def test_cluster_members_read_their_environment_from_the_cluster_record(producer):
    module, data = producer
    _members(data.parent)
    g = module.Group()
    module.environments(g, data.parent)
    v = {k: e["value"] for k, e in g.values.items()}
    assert (v["member4_torch"], v["member4_cuda"], v["member4_numpy"]) == ("2.7.1+cu118", "11.8", "2.4.6")
    assert (v["member5_node"], v["member5_device"], v["member5_slurm_job"]) == ("gpu01", "gpu:a100:2", "1_5")
    assert v["member4_python"] == "3.11.15"
    assert v["member1_environment"] is None


def test_fence_lift_ranges_come_from_the_scoring_runs_fence_report(producer, tmp_path, monkeypatch):
    module, _data = producer
    scored = tmp_path / "scored"
    scored.mkdir()
    (scored / "fence_effect.json").write_text(json.dumps({"summary": {"backbone_pooled": {
        "mae_change_pct_range": [0.8381, 3.1459], "rmse_ratio_range": [1.0111, 1.4387]}}}))
    monkeypatch.setenv("SI_FINAL_SCORES_DIR", str(scored))
    g = module.Group()
    module.fence_effect(g)
    v = {k: e["value"] for k, e in g.values.items()}
    assert v["fence_lift_backbone_mae_change_pct_min"] == pytest.approx(0.8381)
    assert v["fence_lift_backbone_mae_change_pct_max"] == pytest.approx(3.1459)
    assert v["fence_lift_backbone_rmse_ratio_min"] == pytest.approx(1.0111)
    assert v["fence_lift_backbone_rmse_ratio_max"] == pytest.approx(1.4387)


def test_a_copy_threshold_tie_counts_however_the_floats_round():
    """5.39 against 5.395 is exactly 0.005 ppm in the stored decimals; binary subtraction
    lands either side of the threshold, and "within 0.005 ppm" must count it either way."""
    import emit_text_numbers as module
    import numpy as np
    prediction = np.array([5.39, 8.125, 120.5, 4.0])
    target = np.array([5.395, 8.12, 120.4949, 4.0051])
    assert module.within_copy_threshold(prediction, target).tolist() == [True, True, False,
                                                                         False]
