"""Table S8 is written from the edge tables of the build it describes."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SI_BUILD = Path(__file__).resolve().parents[1] / "si_build"

FAMILIES = [
    ("same_bmrb", "labelset_edges_same_bmrb.csv"),
    ("uniprot", "labelset_edges_uniprot.csv"),
    ("exact_token", "labelset_edges_exact_token.csv"),
    ("alignment", "labelset_edges_alignment.csv"),
    ("mmseqs_cov0", "labelset_edges_mmseqs_cov0.csv"),
    ("mmseqs_containment", "labelset_edges_mmseqs_containment.csv"),
    ("modeled_region", "labelset_edges_modeled_region.csv"),
    ("partner_chain", "labelset_edges_partner_chain.csv"),
]
GROUPS_PER_FAMILY = 3


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SI_BUILD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build(root: Path) -> Path:
    """A build whose eight edge tables each hold three groups, none of them crossing."""
    data = root / "data"
    data.mkdir(parents=True)
    (data / "fold_assignments_labelset_identity90.json").write_text(json.dumps(
        {f"ls{index}": index % 7 for index in range(1, 13)}))
    for _family, filename in FAMILIES:
        rows = ["edge_group,fold,label_set_id"]
        for group in range(GROUPS_PER_FAMILY):
            rows += [f"g{group},{group + 1},ls{2 * group + 1}",
                     f"g{group},{group + 1},ls{2 * group + 2}"]
        (data / filename).write_text("\n".join(rows) + "\n")
    return root


def _their_split(root: Path) -> Path:
    """The measurement of the published split this table prints beside ours."""
    artifacts = root / "split_measurement"
    artifacts.mkdir(parents=True)
    (artifacts / "ucb_split_criteria.json").write_text(json.dumps({
        "mmseqs": {"identity": 0.8, "coverage": 0.9, "modeled_region_identity": 0.8,
                   "modeled_region_coverage": 0.9, "modeled_region_coverage_modes": [0, 1]},
        "families": {family: {"groups": 5, "crossing": 1, "test_chains_affected": 2,
                              "train_chains_affected": 3, "test_chains": 2,
                              "train_chains": 3}
                     for family, _filename in FAMILIES},
        "split": {"train_chains_read": 4, "test_chains_read": 2},
    }))
    return artifacts


def test_the_leakage_table_is_written_from_this_builds_edge_tables_whatever_a_vendored_count_says(
        monkeypatch, tmp_path):
    """The group counts printed are the ones this build's edge tables hold. No copy of a
    table produced elsewhere is read, so a partition whose groups moved is a table with
    new numbers in it, not a refusal to write one."""
    run = _build(tmp_path / "run")
    artifacts = _their_split(tmp_path)
    monkeypatch.setenv("DEEPSHIFT_RUN", str(run))
    monkeypatch.setenv("SI_LEAKAGE_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))

    sys.path.insert(0, str(SI_BUILD))
    try:
        si_common = _load("si_common")
        # A directory holding no vendored copy of anything: a generator that reached for
        # one would fail here rather than write the table.
        monkeypatch.setattr(si_common, "REFERENCE_INPUTS", tmp_path / "nothing_vendored")
        emit_leakage = _load("emit_leakage")
        emit_leakage.main()
    finally:
        sys.path.remove(str(SI_BUILD))

    tex = (tmp_path / "out" / "si_leakage.tex").read_text()
    assert tex.count(f"& {GROUPS_PER_FAMILY} & 0 &") == len(FAMILIES)


def test_the_leakage_table_refuses_a_measurement_made_at_another_modelled_region_gate(
        monkeypatch, tmp_path):
    """The modelled-region row states one gate; a split measured at another is refused."""
    run = _build(tmp_path / "run")
    artifacts = _their_split(tmp_path)
    measured = json.loads((artifacts / "ucb_split_criteria.json").read_text())
    measured["mmseqs"].update(modeled_region_identity=0.9, modeled_region_coverage_modes=[0])
    (artifacts / "ucb_split_criteria.json").write_text(json.dumps(measured))
    monkeypatch.setenv("DEEPSHIFT_RUN", str(run))
    monkeypatch.setenv("SI_LEAKAGE_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("SI_OUT", str(tmp_path / "out"))

    sys.path.insert(0, str(SI_BUILD))
    try:
        _load("si_common")
        emit_leakage = _load("emit_leakage")
        with pytest.raises(SystemExit, match="modelled-region gate"):
            emit_leakage.main()
    finally:
        sys.path.remove(str(SI_BUILD))
    assert not (tmp_path / "out" / "si_leakage.tex").exists()
