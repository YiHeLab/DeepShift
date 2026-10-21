"""A rereference offset corrects a whole reference channel, not one atom."""
from deepshift.data.reref_utils import offset_columns


def test_offset_reaches_every_carbon_and_nitrogen_but_no_proton():
    cols = offset_columns({"CA": -0.4, "CB": -0.4, "CO": 0.15, "N": -0.2})
    # side-chain carbons inherit the joint CA/CB aliphatic offset
    assert cols["ca_shift"] == cols["cb_shift"] == -0.4
    assert cols["cg_shift"] == cols["cd1_shift"] == cols["cz_shift"] == -0.4
    # side-chain amide nitrogens inherit the 15N offset
    assert cols["n_shift"] == cols["nd2_shift"] == cols["ne2_shift"] == -0.2
    # C' is calibrated in its own experiment and keeps its own offset
    assert cols["c_shift"] == 0.15
    # protons are never rereferenced (RefDB p.183)
    assert "h_shift" not in cols and "ha_shift" not in cols
    assert not any(col.startswith("h") for col in cols)
    # Every heavy-atom column and nothing else, derived from the vocabulary rather
    # than counted: a fixed count would hold at one vocabulary size while heavy-atom
    # columns added later went uncorrected.
    from deepshift.domain import SHIFT_COLUMNS
    assert set(cols) == {c for c in SHIFT_COLUMNS if c[0] in "cn"}


def test_every_heavy_atom_column_is_rereferenced():
    """The correction applies to a reference channel, so its coverage follows the
    vocabulary rather than a hand-kept list. Literal column tuples written at one
    vocabulary size leave the nuclei added after them -- Trp NE1/CZ2/CZ3/CE3/CH2 and Arg
    NE -- carrying their uncorrected per-deposition constant.

    Asserted against the element classification, so adding a nucleus cannot reopen it.
    """
    from deepshift.data.reref_utils import CO_COL, offset_columns
    from deepshift.domain import SHIFT_COLUMNS

    aliphatic, nitrogen, cprime = 0.5, -0.2, 0.3
    cols = offset_columns({'CA': aliphatic, 'CB': aliphatic, 'CO': cprime, 'N': nitrogen})

    for column in SHIFT_COLUMNS:
        element = column[0]
        if element == 'h':
            assert column not in cols, f'{column}: protons are not corrected'
        elif element == 'n':
            assert cols.get(column) == nitrogen, f'{column}: missing the 15N offset'
        elif element == 'c':
            want = cprime if column == CO_COL else aliphatic
            assert cols.get(column) == want, f'{column}: missing the 13C offset'
        else:
            raise AssertionError(f'{column}: unclassified element')

    heavy = [c for c in SHIFT_COLUMNS if c[0] in 'cn']
    assert len(cols) == len(heavy), f'{len(cols)} corrected vs {len(heavy)} heavy-atom columns'
