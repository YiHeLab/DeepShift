"""The final build admits only sample-matched protein partners."""
import argparse
from pathlib import Path

import pytest

from deepshift.data.build import Builder, add_build_arguments


def _args(tmp_path, **over):
    parser = argparse.ArgumentParser()
    add_build_arguments(parser)
    ns = parser.parse_args(['--root', str(tmp_path)])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _atoms_argv(tmp_path, monkeypatch, **over):
    """Every argv the atoms stage would be spawned with, without running anything."""
    builder = Builder(_args(tmp_path, **over))
    builder.paths.data.mkdir(parents=True, exist_ok=True)
    (builder.paths.data / "atom_census.json").write_text('{"chosen_m_atoms": 512}')
    monkeypatch.setattr(type(builder), 'cache_folds', property(lambda self: [1]))
    seen = []
    monkeypatch.setattr(builder, 'command',
                        lambda name, mod, *a, **kw: seen.append([str(x) for x in a]))
    builder.atoms()
    return seen


def test_every_atom_stage_invocation_carries_the_sample_matched_partner_flags(tmp_path,
                                                                             monkeypatch):
    amap = tmp_path / 'sample_assembly.json'
    amap.write_text('{"label_sets": {}}')
    argv = _atoms_argv(tmp_path, monkeypatch,
                       leaky_partner_chains=str(tmp_path / 'suppress.json'),
                       partner_sample_policy='sample_matched', sample_assembly=str(amap))
    assert argv, 'the atom stage was never invoked'
    for one in argv:
        assert one[one.index('--partner_sample_policy') + 1] == 'sample_matched'
        assert one[one.index('--sample_assembly') + 1] == str(amap)
        assert one[one.index('--leaky_partner_chains') + 1] == str(tmp_path / 'suppress.json')


def test_the_build_parser_has_no_partner_opt_out():
    parser = argparse.ArgumentParser()
    add_build_arguments(parser)
    assert 'multichain' not in {action.dest for action in parser._actions}


def test_all_partner_policy_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="only sample_matched"):
        _atoms_argv(tmp_path, monkeypatch,
                    leaky_partner_chains=str(tmp_path / 'suppress.json'),
                    partner_sample_policy='all')


def test_sample_matched_uses_the_generated_maps_by_default(tmp_path, monkeypatch):
    argv = _atoms_argv(tmp_path, monkeypatch, partner_sample_policy='sample_matched',
                       sample_assembly=None, leaky_partner_chains=None)
    expected = str((tmp_path / "data" / "sample_assembly.json").resolve())
    expected_suppress = str((tmp_path / "data" / "partner_chain_suppression.json").resolve())
    for one in argv:
        assert one[one.index('--sample_assembly') + 1] == expected
        assert one[one.index('--leaky_partner_chains') + 1] == expected_suppress


def test_the_assembly_map_is_an_input_so_a_policy_change_cannot_resume(tmp_path, monkeypatch):
    """Two policies differ only inside the NPYs, so the map must gate resume."""
    amap = tmp_path / 'sample_assembly.json'
    amap.write_text('{"label_sets": {}}')
    builder = Builder(_args(tmp_path, leaky_partner_chains=str(tmp_path / 'suppress.json'),
                            partner_sample_policy='sample_matched', sample_assembly=str(amap)))
    builder.paths.data.mkdir(parents=True, exist_ok=True)
    (builder.paths.data / "atom_census.json").write_text('{"chosen_m_atoms": 512}')
    monkeypatch.setattr(type(builder), 'cache_folds', property(lambda self: [1]))
    seen = []
    monkeypatch.setattr(builder, 'command',
                        lambda name, mod, *a, **kw: seen.append(kw.get('inputs', [])))
    builder.atoms()
    assert seen, 'the atom stage was never invoked'
    for inputs in seen:
        assert Path(amap) in [Path(p) for p in inputs], \
            'the assembly map must gate resume, or a policy switch reuses stale arrays'
