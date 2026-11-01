import json
import os

import cryojax.simulator as cxs
import mmdf
import numpy as np
import pandas as pd
import pytest
from cryojax.constants import (
    EmpiricalScatteringFactorParameters,
    PengScatteringFactorParameters,
)
from cryojax.constants._empirical_tables import (
    _CATALASE_TABLE,
    _EMDB_TABLE,
)
from cryojax.constants._scattering_factor_parameters import _atom_types
from cryojax.io import mmdf_to_atoms
from cryojax.io._chemistry import _AMINO_ACID_BONDS


def test_table_sizes():
    assert len(_CATALASE_TABLE) == 42
    assert len(_EMDB_TABLE) == 96
    assert set(_CATALASE_TABLE) <= set(_EMDB_TABLE)
    for table in (_CATALASE_TABLE, _EMDB_TABLE):
        assert all(len(a) == len(b) == 5 for a, b in table.values())


def test_table_values():
    assert _CATALASE_TABLE["H(C)"] == (
        (-0.2309, 0.503, 0.5916, -0.9023, -0.4652),
        (0.6683, 16.7729, 9.7436, 49.7746, 87.8615),
    )
    assert _EMDB_TABLE["H(C)"] == (
        (-0.1529, 4.135, -4.9181, 1.4351, -1.5408),
        (0.076, 24.7071, 34.3897, 49.3086, 87.991),
    )


ELEMENTS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}


@pytest.fixture
def frame(sample_pdb_path):
    return mmdf.read(sample_pdb_path)


def row(df, residue_id, atom):
    return int(np.flatnonzero((df["residue_id"] == residue_id) & (df["atom"] == atom))[0])


def drop(df, residue_id, atoms):
    keep = ~((df["residue_id"] == residue_id) & df["atom"].isin(atoms))
    return df[keep].reset_index(drop=True)


def parameters(df, table="catalase", **kwargs):
    _, atomic_numbers, atom_chemistry = mmdf_to_atoms(df, loads_chemistry=True, **kwargs)
    return EmpiricalScatteringFactorParameters(atomic_numbers, atom_chemistry, table)


def types(df):
    _, atomic_numbers, atom_chemistry = mmdf_to_atoms(df, loads_chemistry=True)
    return _atom_types(atomic_numbers, atom_chemistry)


def test_residue_types_match_sffit_table():
    path = os.path.join(os.path.dirname(__file__), "data", "residue_atom_types.json")
    with open(path) as f:
        expected = json.load(f)
    for residue, residue_types in expected.items():
        bonds = _AMINO_ACID_BONDS[residue]
        names = sorted({name for bond in bonds for name in bond})
        index = {name: i for i, name in enumerate(names)}
        atom_chemistry = dict(
            residue_names=np.array([residue] * len(names)),
            atom_names=np.array(names),
            bonds=np.array([[index[a], index[b]] for a, b in bonds]),
            missing_bonds=np.array([[index["N"], 6], [index["C"], 7]]),
        )
        atomic_numbers = np.array([ELEMENTS[name[0]] for name in names])
        actual = _atom_types(atomic_numbers, atom_chemistry)
        for name, atom_type in residue_types.items():
            assert actual[index[name]] == atom_type, (residue, name)
        for a, b in bonds:
            if b.startswith("H"):
                assert actual[index[b]] == f"H({a[0]})", (residue, b)


def test_1uao_types(frame):
    actual = types(frame)
    expected = {
        (1, "N"): "N(HHHC)",
        (10, "C"): "C(COO)",
        (10, "O"): "O(C)",
        (10, "OXT"): "O(C)",
        (3, "OD1"): "O(C, carboxyl)",
        (5, "OE2"): "O(C, carboxyl)",
        (4, "N"): "N(CCC)",
        (2, "CA"): "C(HCCN)",
        (2, "OH"): "O(HC)",
        (9, "NE1"): "N(HCC)",
    }
    for (residue_id, name), atom_type in expected.items():
        assert actual[row(frame, residue_id, name)] == atom_type, (residue_id, name)


def test_rows_align(frame):
    empirical = parameters(frame)
    assert empirical.a.shape == empirical.b.shape == (138, 5)


def test_terms_sorted_by_width(frame):
    for table in ("catalase", "emdb"):
        assert np.all(np.diff(parameters(frame, table).b, axis=1) >= 0)


def test_never_drawn_hydrogens(frame):
    empirical = parameters(frame)
    for residue_id, name in [(2, "HH"), (6, "HG1"), (8, "HG1")]:
        assert np.all(empirical.a[row(frame, residue_id, name)] == 0)
    for residue_id, name in [(9, "HE1"), (2, "H"), (1, "H1")]:
        assert np.any(empirical.a[row(frame, residue_id, name)] != 0)


def test_missing_never_drawn_hydrogen_allowed(frame):
    complete = parameters(frame)
    edited = drop(frame, 6, ["HG1"])
    actual = parameters(edited)
    np.testing.assert_array_equal(
        actual.a[row(edited, 6, "OG1")], complete.a[row(frame, 6, "OG1")]
    )
    np.testing.assert_array_equal(
        actual.b[row(edited, 6, "OG1")], complete.b[row(frame, 6, "OG1")]
    )


TYROSINE_RING = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]
TYROSINE_RING_HYDROGENS = ["HD1", "HD2", "HE1", "HE2", "HH"]


@pytest.mark.parametrize(
    "residue_id, cut, removed, not_drawn",
    [
        (2, "CA", ["CB", "HB2", "HB3"] + TYROSINE_RING + TYROSINE_RING_HYDROGENS, ["HA"]),
        (2, "CB", TYROSINE_RING + TYROSINE_RING_HYDROGENS, ["HB2", "HB3"]),
        (5, "CG", ["CD", "OE1", "OE2"], ["HG2", "HG3"]),
    ],
)
def test_truncated_side_chain_hydrogens_not_drawn(
    frame, residue_id, cut, removed, not_drawn
):
    complete = parameters(frame)
    edited = drop(frame, residue_id, removed)
    actual = parameters(edited)
    for name in not_drawn:
        assert np.all(actual.a[row(edited, residue_id, name)] == 0), name
    np.testing.assert_array_equal(
        actual.a[row(edited, residue_id, cut)], complete.a[row(frame, residue_id, cut)]
    )


def test_unselected_hydrogens_raise(frame):
    _, atomic_numbers, atom_chemistry = mmdf_to_atoms(
        frame, loads_chemistry=True, selection_string="not element H"
    )
    with pytest.raises(
        ValueError, match="atom N of residue GLY at index 0 has 0 of its 3 hydrogens"
    ) as error:
        EmpiricalScatteringFactorParameters(atomic_numbers, atom_chemistry)
    assert "not selected" in str(error.value)
    assert "(" not in str(error.value)


def test_chain_break_hydrogens_raise(frame):
    edited = drop(frame, 5, frame.loc[frame["residue_id"] == 5, "atom"].tolist())
    with pytest.raises(
        ValueError,
        match=f"atom N of residue THR at index {row(edited, 6, 'N')} has 1 of its 3 ",
    ) as error:
        parameters(edited)
    assert "chain ends and breaks" in str(error.value)


def test_lengths_must_match(frame):
    _, atomic_numbers, atom_chemistry = mmdf_to_atoms(frame, loads_chemistry=True)
    with pytest.raises(
        ValueError, match="`atomic_numbers` has 137 atoms, but `atom_chemistry` has 138"
    ):
        EmpiricalScatteringFactorParameters(atomic_numbers[:-1], atom_chemistry)


def test_selection_keeps_full_model_parameters(frame):
    complete = parameters(frame)
    selected = parameters(frame, selection_string="resSeq 3 to 8")
    rows = np.flatnonzero(frame["residue_id"].between(3, 8))
    np.testing.assert_array_equal(selected.a, complete.a[rows])
    np.testing.assert_array_equal(selected.b, complete.b[rows])
    kept = frame.iloc[rows].reset_index(drop=True)
    assert np.any(selected.a[row(kept, 3, "H")] != 0)


def test_fallback_to_peng_for_missing_type():
    atomic_numbers = np.array([7, 6, 6, 1, 1])
    atom_chemistry = dict(
        residue_names=np.array(["PRO"] * 5),
        atom_names=np.array(["N", "CA", "CD", "H2", "H3"]),
        bonds=np.array([[0, 1], [0, 2], [0, 3], [0, 4]]),
        missing_bonds=np.zeros((0, 2), dtype=int),
    )
    assert _atom_types(atomic_numbers, atom_chemistry)[0] == "N(HHCC)"
    catalase = EmpiricalScatteringFactorParameters(atomic_numbers, atom_chemistry)
    peng = PengScatteringFactorParameters(np.array([7]))
    np.testing.assert_array_equal(catalase.a[0], peng.a[0])
    np.testing.assert_array_equal(catalase.b[0], peng.b[0])
    emdb = EmpiricalScatteringFactorParameters(
        atomic_numbers, atom_chemistry, "emdb", b_minimum=0.0
    )
    a, b = (np.asarray(x) for x in _EMDB_TABLE["N(HHCC)"])
    np.testing.assert_array_equal(emdb.a[0], a[np.argsort(b, kind="stable")])
    np.testing.assert_array_equal(emdb.b[0], np.sort(b))


def test_no_chemistry_rows_use_peng(frame):
    oxygen = frame[frame["atomic_number"] == 8].iloc[[0]]
    extra = []
    for chain, residue, atom, number in [
        ("B", "MG", "MG", 12),
        ("C", "HOH", "O", 8),
        ("C", "HOH", "H1", 1),
    ]:
        new = oxygen.copy()
        new[["chain", "residue", "atom", "atomic_number"]] = [
            chain,
            residue,
            atom,
            number,
        ]
        extra.append(new)
    edited = pd.concat([frame] + extra, ignore_index=True)
    actual = parameters(edited)
    peng = PengScatteringFactorParameters(np.array([12, 8, 1]))
    np.testing.assert_array_equal(actual.a[138:], peng.a)
    np.testing.assert_array_equal(actual.b[138:], peng.b)


def test_table_keyword(frame):
    ca = row(frame, 2, "CA")
    assert not np.array_equal(parameters(frame).a[ca], parameters(frame, "emdb").a[ca])
    with pytest.raises(ValueError):
        parameters(frame, "peng")


def test_b_minimum(frame):
    _, atomic_numbers, atom_chemistry = mmdf_to_atoms(frame, loads_chemistry=True)
    published = EmpiricalScatteringFactorParameters(
        atomic_numbers, atom_chemistry, b_minimum=0.0
    )
    _, b = _CATALASE_TABLE["C(HCCN)"]
    np.testing.assert_array_equal(published.b[row(frame, 2, "CA")], np.sort(b))
    assert np.min(b) == 0
    floored = EmpiricalScatteringFactorParameters(atomic_numbers, atom_chemistry)
    np.testing.assert_array_equal(floored.a, published.a)
    np.testing.assert_array_equal(floored.b, np.maximum(published.b, 0.2))


def test_volume_from_empirical_parameters(frame):
    positions, atomic_numbers, atom_chemistry = mmdf_to_atoms(frame, loads_chemistry=True)
    empirical = EmpiricalScatteringFactorParameters(atomic_numbers, atom_chemistry)
    volume = cxs.GaussianMixtureVolume.from_tabulated_parameters(positions, empirical)
    assert np.all(np.asarray(volume.variances) > 0)


@pytest.mark.parametrize(
    "residue_id, name, neighbours", [(6, "TPO", [5, 7]), (10, "NME", [9])]
)
def test_bonded_non_standard_residue(frame, residue_id, name, neighbours):
    renamed = frame.copy()
    renamed.loc[renamed["residue_id"] == residue_id, "residue"] = name
    complete, actual = parameters(frame), parameters(renamed)
    rows = np.flatnonzero(frame["residue_id"].isin(neighbours))
    np.testing.assert_array_equal(actual.a[rows], complete.a[rows])
    np.testing.assert_array_equal(actual.b[rows], complete.b[rows])
    own = np.flatnonzero(frame["residue_id"] == residue_id)
    peng = PengScatteringFactorParameters(frame["atomic_number"].to_numpy()[own])
    np.testing.assert_array_equal(actual.a[own], peng.a)
