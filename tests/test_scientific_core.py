"""The package owns one implementation of each shared scientific transform."""

import pytest

from deepshift import chemistry, domain


def test_chemistry_exports_are_the_domain_vocabulary():
    for name in (
        "AA_3_TO_1",
        "ATOM_TO_IDX",
        "ATOM_TYPES",
        "NUCLEI",
        "RESIDUE_TO_IDX",
        "SHIFT_COLUMNS",
        "STANDARD_RESIDUES",
    ):
        assert getattr(chemistry, name) is getattr(domain, name)


@pytest.mark.parametrize("column", domain.SHIFT_COLUMNS)
def test_shift_column_to_atom_is_exact(column):
    assert chemistry.atom_from_shift_column(column) == column.removesuffix("_shift").upper()


@pytest.mark.parametrize("column", ["", "ca", "_shift", "ca_shift_extra"])
def test_invalid_shift_column_is_refused(column):
    with pytest.raises(ValueError, match="not a shift column"):
        chemistry.atom_from_shift_column(column)


def test_the_trainer_and_the_atom_stage_require_one_atom_vocabulary():
    """The cache the atom stage stamps and the recipe the trainer requires of it are the
    same object, so a fold cannot be trained on a vocabulary the stage would not write."""
    import inspect

    from deepshift import training
    from deepshift.data import atoms

    assert "shipped_atom_cloud_recipe()" in inspect.getsource(training.load_atom_recipe)
    assert atoms.atom_cloud_recipe.__module__ == atoms.shipped_atom_cloud_recipe.__module__
