"""The case-study hold-out's second rule and its boundary report."""
from deepshift.data import folds


def _universe(sequences):
    return {ls: {"label_set_id": ls, "bmrb_id": bmrb, "entity_sequence": seq}
            for ls, (bmrb, seq) in sequences.items()}


def test_a_long_near_identical_segment_is_matched_whatever_its_coverage():
    ls_rec = _universe({"frag": ("52694", "A" * 145), "far": ("1", "G" * 50)})
    hits = {("__case__tau_k18", "frag"): {"identity": 0.987, "aligned": 76,
                                          "query_coverage": 0.59, "target_coverage": 0.52},
            ("__case__tau_k18", "far"): {"identity": 0.95, "aligned": 12,
                                         "query_coverage": 0.1, "target_coverage": 0.2}}
    matched = folds.case_study_segment_matched_labelsets(
        ls_rec, {"tau_k18": "K" * 129}, search_fn=lambda q, t: hits)
    assert matched == {"frag": {"case": "tau_k18", "identity": 0.987, "aligned": 76}}


def test_the_gate_catches_a_rule_match_left_in_training_not_a_transitive_group():
    ls_rec = _universe({"seed": ("25500", "M" * 59), "chained": ("7", "G" * 80),
                        "missed": ("8", "M" * 59)})
    assignment = {"seed": 0, "chained": 4, "missed": 2}
    families = {"partner_chain": [["seed", "chained"]], "alignment": [["seed", "missed"]]}
    # "missed" satisfies the rule (same deposition family) but was left in fold 2
    report = folds.case_study_boundary({"seed"}, {"seed": {"same_bmrb"}}, {"seed", "missed"},
                                       assignment, ls_rec, families)
    assert set(report["prohibited"]) == {"missed"}
    # the transitive partner-chain relative is reported, never gated
    assert set(report["retained_training_relatives"]) == {"partner_chain", "alignment"}
    assert report["retained_training_relatives"]["partner_chain"]["chained"]["fold"] == 4
    # once it is pinned too, nothing is prohibited though the groups still span folds
    assignment["missed"] = 0
    clean = folds.case_study_boundary({"seed", "missed"}, {}, {"seed", "missed"},
                                      assignment, ls_rec, families)
    assert clean["prohibited"] == {}


def test_the_calibration_exclusion_covers_relatives_and_the_held_out_proteins():
    assignment = {"rel": 0, "case": 0, "case_elsewhere": 2, "clean": 0}
    record = folds.calibration_exclusion_record(["rel"], {"case", "case_elsewhere"}, assignment)
    # only the case-study records that landed in fold 0 can pollute a fold-0 calibration
    assert record["label_sets"] == ["case", "rel"]
    assert record["reasons"] == {"training_fold_relative": 1, "case_study_holdout": 1, "both": 0}


def test_the_boundary_summary_counts_the_refused_label_sets_rather_than_summing_them():
    boundary = {'retained_training_relatives': {'partner_chain': {'a': {}, 'b': {}}},
                'prohibited': {'ls1': {'bmrb': '25500', 'fold': 2},
                               'ls2': {'bmrb': '19253', 'fold': 4}}}
    line = folds.case_study_boundary_summary(boundary)
    assert 'prohibited 2' in line
    assert '{"partner_chain": 2}' in line
    assert 'prohibited 0' in folds.case_study_boundary_summary(
        {'retained_training_relatives': {}, 'prohibited': {}})
