"""One test per property whose violation is silent.

Not coverage. Each of these pins one boundary: a fold map that moves under a rebuild,
a homolog that spans two folds, a QC fence that matches nothing, a validation carve
that withholds rows from every model, and a benchmark protein whose sub-threshold
homolog sits in training. Two of them live in ``test_training.py`` -- see the cross-references below --
because they need the trainer's own fixture.

    QC fence matching zero cells raises
        -> test_training.py::test_fence_matching_no_cell_fails_loudly
    'final' selection withholds exactly zero rows
        -> test_training.py::test_final_selection_withholds_exactly_zero_rows
"""
import importlib.resources
import json
from pathlib import Path

import pytest

from deepshift.data import cluster, folds

N_COMPONENTS = 30


def _components(n):
    return {f"ls_{i:04d}": [f"ls_{i:04d}"] for i in range(n)}


def _records(n):
    return {f"ls_{i:04d}": {"bmrb_id": f"{17000 + i}", "n_shift_residues": 50 + i,
                            "sources": ["af"]} for i in range(n)}


def test_installed_package_has_only_current_json_authorities():
    data = importlib.resources.files("deepshift.data")
    assert data.joinpath("hetatm_exclusion_rule.json").is_file()
    assert data.joinpath("ucbshift200_excluded_bmrbs.json").is_file()
    assert not data.joinpath("model_cutoffs_p100.json").is_file()


# ---------------------------------------------------------------------------
# 1. Fold assignments must survive a growing dataset
# ---------------------------------------------------------------------------

def test_fold_assignment_is_reproducible_for_one_input():
    """The floor the current algorithm does meet: identical inputs, identical map."""
    first = folds.assign_labelset_folds(_components(N_COMPONENTS),
                                        _records(N_COMPONENTS), set())
    second = folds.assign_labelset_folds(_components(N_COMPONENTS),
                                         _records(N_COMPONENTS), set())
    assert first == second


# ---------------------------------------------------------------------------
# 2. No homolog edge may span two cross-validation folds
# ---------------------------------------------------------------------------

def test_homolog_edge_thresholds_are_the_agreed_axis():
    """identity >= 0.80 and coverage >= 0.90. The transposed pair (0.90/0.80) is a
    different, weaker rule."""
    assert cluster.MMSEQS_IDENTITY == 0.80
    assert cluster.MMSEQS_COVERAGE == 0.90


def _edge_universe():
    universe = {f"le_{tag}": {"label_set_id": f"ls_{tag}", "bmrb_id": bmrb}
                for tag, bmrb in (("a", "1"), ("b", "2"), ("c", "3"))}
    # One synthetic homolog hit above threshold, injected as data -- no MMseqs run.
    edges = folds.build_labelset_edges(universe, {"mmseqs_cov0": [["le_a", "le_b"]]}, {})
    return universe, edges


def test_a_homolog_pair_is_placed_in_one_fold():
    """Components, not label sets, are dealt to folds, so an above-threshold pair cannot
    land on opposite sides of the train/eval boundary."""
    universe, edge_families = _edge_universe()
    _, canon_to_members = folds.labelset_components(universe, edge_families)
    records = {"ls_a": {"bmrb_id": "1", "n_shift_residues": 90, "sources": ["af"]},
               "ls_b": {"bmrb_id": "2", "n_shift_residues": 80, "sources": ["nmr"]},
               "ls_c": {"bmrb_id": "3", "n_shift_residues": 70, "sources": ["xtal"]}}
    assigned = folds.assign_labelset_folds(canon_to_members, records, set())
    assert assigned["ls_a"] == assigned["ls_b"]
    assert folds.verify_no_labelset_leakage(assigned, edge_families) == \
        dict.fromkeys(edge_families, 0)


def test_a_cross_fold_homolog_edge_is_detected():
    """The checker must actually catch a leak; a checker that always returns zero is
    worse than none."""
    _, edge_families = _edge_universe()
    leaked = {"ls_a": 1, "ls_b": 2, "ls_c": 1}
    report = folds.verify_no_labelset_leakage(leaked, edge_families)
    assert report["mmseqs_cov0"] == 1
    assert any(count for count in report.values())


def _containment_search(seqs, *, identity, coverage, cov_mode, binary=None):
    """MMseqs' pair contract on exact-substring pairs: identity 1.0; coverage mode 0
    needs the floor on both sequences, mode 1 on the target of either ordering."""
    pairs = set()
    for left, right in ((a, b) for a in seqs for b in seqs if a < b):
        short, long_ = sorted((seqs[left], seqs[right]), key=len)
        if short not in long_:
            continue
        ratio = len(short) / len(long_)
        if (ratio >= coverage) if cov_mode == 0 else True:
            pairs.add((left, right))
    return pairs


def test_a_modeled_region_contained_in_a_longer_one_lands_in_its_fold(monkeypatch):
    """A 50-residue modeled region that is an exact subsequence of a 300-residue one
    covers 17% of the long region, so the bidirectional mode alone finds no edge; the
    containment mode does. The modeled-region family runs both, like the entity families,
    and the two label sets share a fold."""
    from deepshift.data import sequence_clustering

    long_region = "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(300))
    short_region = long_region[100:150]
    calls = []

    def search(seqs, *, identity, coverage, cov_mode, binary=None):
        calls.append(cov_mode)
        return _containment_search(seqs, identity=identity, coverage=coverage,
                                   cov_mode=cov_mode)

    monkeypatch.setattr(sequence_clustering, "run_all_vs_all", search)
    universe = {
        "le_long": {"label_set_id": "ls_long", "bmrb_id": "1", "n_shift_residues": 300,
                    "sources": ["nmr"], "modeled_by_source": {"nmr": long_region}},
        "le_short": {"label_set_id": "ls_short", "bmrb_id": "2", "n_shift_residues": 50,
                     "sources": ["nmr"], "modeled_by_source": {"nmr": short_region}},
    }
    groups = folds.source_specific_modeled_groups(universe)
    assert groups == [["le_long", "le_short"]]
    assert set(calls) == set(cluster.MMSEQS_COVERAGE_MODES) == {0, 1}

    assigned, families, _, _ = folds.assign_folds_direct_exclusion(
        universe, {}, {}, groups, excluded_seed=set())
    assert assigned["ls_long"] == assigned["ls_short"]
    assert families["modeled_region"] == [["ls_long", "ls_short"]]


def test_a_fold0_label_set_with_a_training_fold_twin_is_counted_and_listed():
    """Fold 0 is the interval-calibration set. A fold-0 label set whose exact-sequence
    twin trains a member would make the calibration residuals optimistic, so the folds
    stage counts such edge groups per family and lists the fold-0 members for the
    interval fit to exclude."""
    universe = {f"le_{i}": {"label_set_id": f"ls_{i}", "bmrb_id": str(100 + i)}
                for i in range(1, 13)}
    families = folds.build_labelset_edges(
        universe, {"exact_token": {"g1": ["le_1", "le_2"]}}, {})
    assignment = {f"ls_{i}": 1 + (i % 5) for i in range(1, 13)}
    assignment.update(ls_1=folds.EXCLUDED_FOLD, ls_2=3)
    counts, relatives = folds.fold0_training_relatives(assignment, families)
    assert counts["exact_token"] == 1
    assert sum(counts.values()) == 1
    assert relatives == ["ls_1"]

    # A twin in the reserved evaluation fold trains no member and is not a relative.
    assignment["ls_2"] = folds.HOLDOUT_FOLD
    counts, relatives = folds.fold0_training_relatives(assignment, families)
    assert counts["exact_token"] == 0 and relatives == []


# ---------------------------------------------------------------------------
# 5. A checkpoint and its eval set must resolve to one fold map
# ---------------------------------------------------------------------------

def test_one_bmrb_spanning_two_folds_is_reported_not_collapsed():
    """A BMRB-keyed map used against label-set-keyed evals overlaps train and eval.
    Projecting label sets onto BMRB ids is lossy, and the projection must say so and
    never silently pick a fold."""
    records = {"ls_a": {"bmrb_id": "700"}, "ls_b": {"bmrb_id": "700"},
               "ls_c": {"bmrb_id": "701"}}
    agreeing, agree, conflicts = folds.derived_bmrb_fold_map(
        {"ls_a": 3, "ls_b": 3, "ls_c": 4}, records)
    assert agreeing == {"700": 3, "701": 4} and agree and not conflicts

    _, agree, conflicts = folds.derived_bmrb_fold_map(
        {"ls_a": 3, "ls_b": 4, "ls_c": 4}, records)
    assert not agree
    assert conflicts == {"700": [3, 4]}


# ---------------------------------------------------------------------------
# 6. Fold 0 must hold the whole benchmark boundary, not the CV axis
# ---------------------------------------------------------------------------

def test_the_fold0_homology_gate_is_its_own_wider_axis():
    """Fold 0 excludes homologs of a published benchmark; folds 1-6 only group twins.
    One threshold pair shared between them conflates two different questions.

    At the CV axis (0.80/0.90), ``1PGX_``, ``1UWXA`` and ``1TCF_`` each keep a
    sub-threshold homolog in a training fold whose proton shifts transfer -- ten of
    them for ``1TCF_``, one EF-hand family in fold 4. The fold-0 arm therefore carries
    its own thresholds, below the CV axis; raising them to the CV axis reopens the leak,
    and lowering the CV axis to meet them coarsens every training fold instead."""
    assert (folds.UCB_HOMOLOGY_IDENTITY, folds.UCB_HOMOLOGY_COVERAGE) == (0.65, 0.60)
    assert folds.UCB_HOMOLOGY_IDENTITY < cluster.MMSEQS_IDENTITY
    assert folds.UCB_HOMOLOGY_COVERAGE < cluster.MMSEQS_COVERAGE


def test_the_fold0_homology_search_runs_at_the_fold0_thresholds(monkeypatch):
    """A named constant nobody reaches is not a gate. The search is asked for the
    fold-0 thresholds, over both coverage modes.

    Both halves are load-bearing. The thresholds go into MMseqs' prefilter, so a call
    left at the CV axis silently keeps the narrower boundary. And the coverage term
    means 'the shorter side', realized as the union of cov-mode 1 and 2 -- one mode
    alone misses whichever side is the fragment, and one real partner is a fragment
    (``1PGX_``'s aligns at 0.66 coverage of the benchmark sequence)."""
    calls = []

    def record(seqs, *, identity, coverage, cov_mode, binary=None):
        calls.append({"identity": identity, "coverage": coverage, "cov_mode": cov_mode})
        return {("__ucbtest__1TCF_", "ls_partner")}

    monkeypatch.setattr("deepshift.data.sequence_clustering.run_all_vs_all", record)
    matched = folds.ucb_homology_matched_labelsets(
        {"ls_partner": {"entity_sequence": "MDDIYKAAVEQLTEEQKNEFKAAFDIFVLGAEDGSIS"}},
        {"1TCF_": "ASGEHVTDEEIESLMKDGDKNNDGRIDFDEFLKMMEGVQ"})

    assert matched == {"ls_partner"}
    assert {call["identity"] for call in calls} == {folds.UCB_HOMOLOGY_IDENTITY}
    assert {call["coverage"] for call in calls} == {folds.UCB_HOMOLOGY_COVERAGE}
    assert {call["cov_mode"] for call in calls} == set(folds.UCB_HOMOLOGY_COVERAGE_MODES) == {1, 2}


def test_every_benchmark_protein_is_on_the_exclusion_authority():
    """Every one of the 200 benchmark proteins must be on the exclusion authority.

    A list generated by scraping a coordinate directory cannot notice that a benchmark
    member is absent, and for a missing member the same-BMRB and same-UniProt arms never
    fire: it reaches fold 0 only if the homology arm matches its sequence against itself,
    and a training label set can keep a UniProt that should have excluded it. BMRB 1553
    (``1TCF_``), 5954 (``1KTZB``) and 7197 (``3DFR_``) are checked by name."""
    authority = folds._packaged_ucb_excluded_bmrbs()
    declared = json.loads(Path(folds.__file__)
                          .with_name("ucbshift200_excluded_bmrbs.json").read_text())
    benchmark = declared.get("test_200_bmrbs", [])
    assert len(benchmark) == 200
    assert set(benchmark) <= authority
    assert {"1553", "5954", "7197"} <= authority


def test_an_authority_missing_a_benchmark_protein_is_refused(tmp_path):
    """Restating the 200 next to the exclusion list is only worth something if a
    disagreement is fatal. A regenerated list that drops a benchmark member must fail
    the load, never be trusted."""
    path = tmp_path / "ucbshift200_excluded_bmrbs.json"
    path.write_text(json.dumps({"bmrbs": ["1552", "1554"],
                                "test_200_bmrbs": ["1552", "1553"]}))
    with pytest.raises(ValueError, match="1553"):
        folds.load_shipped_excluded_bmrbs(str(path))


def _lumpy_universe(n_components=200, big=40):
    """Components of uneven size, as real leakage components are: one holds ``big``
    label sets, the rest are singletons. Returns (canon_to_members, records)."""
    members = {"ls_big": [f"ls_big_{i:03d}" for i in range(big)]}
    members.update({f"ls_{i:04d}": [f"ls_{i:04d}"] for i in range(n_components - 1)})
    records = {ls: {"bmrb_id": f"b{ls}", "n_shift_residues": 60, "sources": ["af"]}
               for group in members.values() for ls in group}
    return members, records


def test_fold0_exclusion_wins_over_the_fold6_holdout_draw():
    """Ordering. A UCB-200-related label set lands in fold 0, never in the holdout.

    Exclusion is applied before the holdout is drawn -- twice over: the direct-exclusion
    path drops the seed from the universe before the component graph is built, and
    ``assign_labelset_folds`` stamps excluded components before it shuffles a manifest
    that does not contain them. Neither gate can take a label set from the other, which
    matters because a holdout holding a benchmark homolog is silent: fold 6 is an eval
    fold, so nothing downstream complains."""
    canon_to_members, records = _lumpy_universe()
    excluded = {"ls_big"} | {f"ls_{i:04d}" for i in range(30)}
    assigned = folds.assign_labelset_folds(canon_to_members, records, excluded)

    for canon in excluded:
        for ls in canon_to_members[canon]:
            assert assigned[ls] == folds.EXCLUDED_FOLD
    assert not any(assigned[ls] == folds.HOLDOUT_FOLD
                   for canon in excluded for ls in canon_to_members[canon])


def test_the_fold6_holdout_reaches_its_target_after_exclusion_takes_its_pick():
    """Widening the fold-0 gate takes label sets the holdout would otherwise hold, and
    the holdout gate refuses rather than redrawing.

    A draw that takes a fixed fraction of components cannot hold a band the gate measures
    in label sets, BMRBs and residue mass: components are heavy-tailed, one of them can
    carry several percent of the label sets on its own, and the band then rides on the
    seed. Targeting the label-set fraction directly holds fold 6 at ~10% whatever
    exclusion takes first."""
    canon_to_members, records = _lumpy_universe()
    singletons = {f"ls_{i:04d}" for i in range(30)}
    with_big = {"ls_big"} | singletons
    eligible = len(records) - sum(len(canon_to_members[c]) for c in with_big)

    for excluded in (set(), singletons, with_big):
        assigned = folds.assign_labelset_folds(canon_to_members, records, excluded)
        gate = folds.check_holdout_gate(assigned, records)
        assert gate["passed"], f"excluded={len(excluded)}: {gate['reason']}"
        assert folds.HOLDOUT_MIN_FRAC <= gate["fractions"]["labelset"] \
            <= folds.HOLDOUT_MAX_FRAC

    # ...and the holdout is drawn from what survives exclusion, so it neither shrinks
    # with the excluded pool nor borrows from it.
    assigned = folds.assign_labelset_folds(canon_to_members, records, with_big)
    held = sum(1 for f in assigned.values() if f == folds.HOLDOUT_FOLD)
    assert round(folds.HOLDOUT_FRAC * eligible) == held


def test_a_holdout_gate_failure_says_what_to_do_about_it():
    """A gate that stops the build without a diagnosis leaves the operator nothing to act
    on. A fold assignment that stops here tells the operator that the draw is
    deterministic (so a rerun reproduces the miss), which dimension missed, and which two
    remedies are legitimate -- not 'try seeds until it passes', which selects the split
    on the outcome the gate audits."""
    records = _records(N_COMPONENTS)
    # A lopsided holdout: 1 of 30 label sets, far under the 9% floor.
    lopsided = {ls: (folds.HOLDOUT_FOLD if i == 0 else 1)
                for i, ls in enumerate(sorted(records))}
    gate = folds.check_holdout_gate(lopsided, records)
    assert not gate["passed"]
    assert gate["fractions"]["labelset"] == pytest.approx(1 / 30)
    assert "labelset holdout frac" in gate["reason"]
    assert f"[{folds.HOLDOUT_MIN_FRAC:.2f},{folds.HOLDOUT_MAX_FRAC:.2f}]" in \
        gate["reason"]

    remedy = folds.holdout_gate_remedy(gate)
    assert "DETERMINISTIC" in remedy and "SHIFT_SPLIT_SEED" in remedy
    assert "SHIFT_HOLDOUT_MIN_FRAC" in remedy      # the second legitimate remedy
    assert "Do NOT search seeds" in remedy         # and the illegitimate one, named
    assert gate["reason"] in remedy     # names the dimension


def test_sharing_a_uniprot_with_a_benchmark_protein_excludes_a_training_label_set():
    """The accession arm, exercised directly: a label set that shares a validated
    accession with a benchmark protein is seeded to fold 0 even with no sequence match
    at all. On the live lineage this arm alone moves BMRB 17276 (accession P01137,
    shared with ``1KTZB``), because no MMseqs alignment between the two exists for the
    homology arm to see."""
    records = {"ls_sibling": {"bmrb_id": "17276"}, "ls_unrelated": {"bmrb_id": "18000"}}
    seeded = folds.direct_ucb_excluded_labelsets(
        records, test_bmrbs={"5954"}, test_uniprots={"P01137"},
        ls_uniprots={"ls_sibling": {"P01137"}, "ls_unrelated": {"P99999"}},
        homology_matched=set())
    assert seeded == {"ls_sibling"}


def test_ucb_target_authority_is_complete_json_and_a_pickle_is_refused(tmp_path):
    targets = {f"T{i:03d}": "ACDEFGHIK" for i in range(200)}
    json_path = tmp_path / "test_200.json"
    json_path.write_text(json.dumps(targets))
    assert folds.load_ucb_test_target_sequences(json_path) == targets

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"one": "ACD"}))
    with pytest.raises(ValueError, match="exactly 200"):
        folds.load_ucb_test_target_sequences(incomplete)

    pickle_path = tmp_path / "test_200.pkl"
    pickle_path.write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="must be a JSON file"):
        folds.load_ucb_test_target_sequences(pickle_path)


def test_fold_verify_reads_and_compares_the_published_artifacts(tmp_path):
    records = {
        "ls_a": {
            "logical_entity_id": "le_a", "label_set_id": "ls_a",
            "bmrb_id": "1", "entity_id": "1", "modeled_length": 12,
            "entity_length": 12, "n_shift_residues": 12, "sources": ["af"],
        },
        "ls_b": {
            "logical_entity_id": "le_b", "label_set_id": "ls_b",
            "bmrb_id": "2", "entity_id": "1", "modeled_length": 14,
            "entity_length": 14, "n_shift_residues": 14, "sources": ["nmr"],
        },
    }
    assignment = {"ls_a": 1, "ls_b": 2}
    bmrb_map = {"1": 1, "2": 2}
    balance = folds.check_labelset_balance(assignment, records, tol=10.0)
    edge_families = {family: [] for family in folds.LEAKAGE_FAMILIES}
    edge_families["exact_token"] = [["ls_a", "ls_b"]]
    extra = {"leaks_per_family": dict.fromkeys(folds.LEAKAGE_FAMILIES, 0)}

    (tmp_path / Path(folds.OUT_LABELSET_FOLD_MAP).name).write_text(
        json.dumps(assignment, sort_keys=True)
    )
    (tmp_path / Path(folds.OUT_LABELSET_BMRB_MAP).name).write_text(
        json.dumps(bmrb_map, sort_keys=True)
    )
    (tmp_path / Path(folds.OUT_BALANCE_REPORT).name).write_text(
        json.dumps(balance, sort_keys=True)
    )
    (tmp_path / Path(folds.OUT_QUARANTINE).name).write_text(
        "logical_entity_id,label_set_id,bmrb_id,entity_id,"
        "modeled_length,entity_length\n"
    )
    folds.write_labelset_edge_tables(
        edge_families, records, assignment, data_dir=tmp_path,
        extra_provenance=extra,
    )
    relatives = folds.fold0_training_relatives(assignment, edge_families)
    (tmp_path / Path(folds.OUT_FOLD0_RELATIVES).name).write_text(json.dumps(
        folds.fold0_relatives_record(*relatives), sort_keys=True))
    assert folds.verify_published_entity_folds(
        tmp_path, assignment, bmrb_map, [], balance, edge_families, records,
        extra_provenance=extra, fold0_relatives=relatives,
    )

    # Verification reads the published file itself: a mutated map on disk is caught
    # even when every gate run against the fresh in-memory assignment is clean.
    (tmp_path / Path(folds.OUT_LABELSET_FOLD_MAP).name).write_text(
        json.dumps({"ls_a": 2, "ls_b": 1}, sort_keys=True)
    )
    with pytest.raises(RuntimeError, match="differs from fresh fold assignment"):
        folds.verify_published_entity_folds(
            tmp_path, assignment, bmrb_map, [], balance, edge_families, records,
            extra_provenance=extra,
        )


# ---------------------------------------------------------------------------
# 6. Sequence alignment must survive a Biopython upgrade
# ---------------------------------------------------------------------------

def test_frame_safe_alignment_forbids_only_the_wrong_gap_direction():
    """The aligner's gap policy is asserted as a constraint rather than through
    Biopython's scoring attribute names, which come and go across releases and take
    every structure pairing with them mid-build: a shorter target aligns by gapping the
    target, and the frame-unsafe direction is dominated."""
    from deepshift.data.alignment import align_sequences, frame_safe_alignments

    short, long_ = "ACDEFGHIK", "ACDEWFGHIK"
    best = frame_safe_alignments(short, long_)[0]
    target, query = str(best[0]), str(best[1])
    assert "-" in target and "-" not in query, "the gap must land in the shorter target"
    assert len(target) == len(query) == len(long_)

    # Equal length: neither direction is allowed, so the alignment is ungapped.
    same = frame_safe_alignments("ACDEFG", "ACDWFG")[0]
    assert "-" not in str(same[0]) and "-" not in str(same[1])

    assert align_sequences(short, long_) is not None


# ---------------------------------------------------------------------------
# 7. The nucleus vocabulary must have exactly one definition
# ---------------------------------------------------------------------------

def test_every_stage_agrees_on_the_nucleus_vocabulary():
    """`fetch.TARGET_ATOMS` and `domain.NUCLEI` are bound to each other, so a nucleus
    added to one cannot be missing from the other. A second hardcoded list would let an
    extension of `domain.NUCLEI` produce a shift table and a cache at the narrower
    width, never reaching the labels and never raising.

    The cache column list is data-driven (`parse_shift_columns` reads whatever
    `*_shift` columns the source has), so `fetch` is what decides them: it is the only
    stage that turns BMRB atom names into columns.
    """
    from deepshift.data.cache_columns import parse_shift_columns
    from deepshift.data.fetch import TARGET_ATOMS
    from deepshift.domain import NUCLEI, SHIFT_COLUMNS

    assert len(NUCLEI) == len(set(NUCLEI)), "NUCLEI has duplicates"
    assert len(TARGET_ATOMS) == len(NUCLEI)
    assert {f"{atom}_shift" for atom in TARGET_ATOMS} == set(SHIFT_COLUMNS)

    # A source header carrying exactly fetch's columns must parse back to the vocabulary,
    # in the sorted order the cache stores.
    header = ["label_set_id", "comp_id"] + [f"{a}_shift" for a in sorted(TARGET_ATOMS)]
    assert parse_shift_columns(header) == sorted(SHIFT_COLUMNS)


# ---------------------------------------------------------------------------
# 8. Wide sparse CSVs must not be read with per-chunk type inference
# ---------------------------------------------------------------------------

def test_shift_table_reads_disable_chunked_type_inference():
    """pandas' C parser segfaults reading the 64-column shift table under the default
    `low_memory=True`. It infers dtypes per chunk, and the sparsest nuclei (Tyr HH has
    523 observed cells in 1.5M rows) are empty for early chunks and populated later, so
    the inferred dtype changes mid-file.

    Measured: `low_memory=True` -> SIGSEGV, `low_memory=False` -> 1,509,465 rows x 140
    columns. It is a correctness requirement, not a tuning knob, so every read of a
    shift-bearing table declares it -- a crash class cannot be caught by a unit test on
    small fixtures, only by pinning the call sites.
    """
    import re
    from pathlib import Path

    import deepshift

    root = Path(deepshift.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        for match in re.finditer(r"read_csv\(", source):
            index, depth = match.end(), 1
            while index < len(source) and depth:
                depth += (source[index] == "(") - (source[index] == ")")
                index += 1
            call = re.sub(r"\s+", " ", source[match.start():index])
            # A shift-bearing table is one keyed by label_set_id; nrows=0 header probes and
            # dtype=str reads do no per-chunk inference and are safe.
            if "label_set_id" not in call or "dtype=str" in call or "nrows=0" in call:
                continue
            if "low_memory=False" not in call:
                offenders.append(f"{path.name}:{source[:match.start()].count(chr(10)) + 1}")
    assert not offenders, ("shift-table read_csv without low_memory=False (segfault risk): "
                           + ", ".join(offenders))


def test_mmseqs_searches_pin_the_evalue_wide_open():
    """A homolog search left at the default E-value reports too few cross-fold edges.

    ``easy-search`` defaults to ``-e 1e-3``. The criteria here are identity and coverage;
    at that default the E-value silently becomes a third one. A short peptide contained in
    a longer sequence meets ``--cov-mode 1 -c 0.9`` only in the long->short direction, and
    that direction scores just above 1e-3 in a database of thousands, so exact-subsequence
    pairs are dropped and the cross-fold edges they carry go unreported.

    Asserting on the constructed argv, not on mmseqs output, keeps this runnable without
    the binary; the mechanism is the flag, so the flag is what is pinned.
    """
    from deepshift.data import sequence_clustering as sc

    captured = []

    class _Result:
        returncode = 0
        stdout = stderr = ""

    def _capture(command, **kwargs):
        captured.append(command)
        raise FileNotFoundError("stop after argv capture")

    for runner, kwargs in ((sc.run_all_vs_all, {}),):
        captured.clear()
        original = sc.subprocess.run
        sc.subprocess.run = _capture
        try:
            runner({"le_a": "GIVEQCCTSICSLYQLENYCN"}, binary="/nonexistent/mmseqs", **kwargs)
        except FileNotFoundError:
            pass
        finally:
            sc.subprocess.run = original
        assert captured, f"{runner.__name__} did not invoke mmseqs"
        argv = captured[0]
        after = {flag: argv[argv.index(flag) + 1] for flag in
                 ("--min-seq-id", "-c", "--cov-mode", "--max-seqs", "--max-accept", "-e")}
        assert argv[1] == "easy-search" and argv[2] == argv[3], "an all-versus-all search"
        assert after == {"--min-seq-id": "0.8", "-c": "0.9", "--cov-mode": "0",
                         "--max-seqs": "1", "--max-accept": "1", "-e": "10000"}


def test_the_holdout_source_tolerance_is_its_own_parameter():
    """The smallest structure source sits in the clumpiest leakage components, so its
    fold-6 share varies far more from seed to seed than the other sources'. The gate
    judges that share against a tolerance of its own, set beside the CV balance
    tolerance and named in the remedy, so a draw that is in band on every mass dimension
    is not refused by a criterion the draw cannot be expected to meet."""
    records = {}
    for i in range(200):
        records[f"ls_{i:04d}"] = {"bmrb_id": f"{17000 + i}", "n_shift_residues": 50,
                                  "sources": ["xtal" if i < 40 else "nmr"]}
    # Hold out 20 of 200 label sets: 5 of the 40 crystal ones (12.5%, a quarter above the
    # 10% overall share) and 15 of the 160 NMR ones (9.4%).
    held = [f"ls_{i:04d}" for i in range(5)] + [f"ls_{i:04d}" for i in range(40, 55)]
    assigned = {ls: (folds.HOLDOUT_FOLD if ls in held else 1) for ls in records}

    strict = folds.check_holdout_gate(assigned, records, tol=0.15)
    assert all(strict["in_band"].values())
    assert strict["source_dev"]["xtal"] == pytest.approx(0.25)
    assert not strict["passed"] and "source xtal" in strict["reason"]
    assert "--holdout_source_tol" in folds.holdout_gate_remedy(strict)

    wide = folds.check_holdout_gate(assigned, records, tol=0.30)
    assert wide["passed"] and wide["tol"] == 0.30
