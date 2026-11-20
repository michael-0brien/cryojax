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
from cryojax.constants._empirical_tables import _CATALASE_TABLE
from cryojax.constants._scattering_factor_parameters import _atom_types
from cryojax.io import mmdf_to_atoms, mmdf_to_topology
from cryojax.io._chemistry import _AMINO_ACID_BONDS


def test_table_size():
    assert len(_CATALASE_TABLE) == 42
    assert all(len(a) == len(b) == 5 for a, b in _CATALASE_TABLE.values())


def test_table_values():
    assert _CATALASE_TABLE["H(C)"] == (
        (-0.2309, 0.503, 0.5916, -0.9023, -0.4652),
        (0.6683, 16.7729, 9.7436, 49.7746, 87.8615),
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
            unselected_bonds=np.zeros((0, 2), dtype=int),
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
    assert np.all(np.diff(parameters(frame).b, axis=1) >= 0)


def test_hydrogens_in_the_model_are_drawn(frame):
    empirical = parameters(frame)
    for residue_id, name in [(2, "HH"), (6, "HG1"), (9, "HE1"), (2, "H"), (1, "H1")]:
        assert np.any(empirical.a[row(frame, residue_id, name)] != 0)


def test_an_absent_hydrogen_changes_its_atoms_type(frame):
    edited = drop(frame, 6, ["HG1"])
    assert types(frame)[row(frame, 6, "OG1")] == "O(HC)"
    assert types(edited)[row(edited, 6, "OG1")] == "O(C)"


TYROSINE_RING = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]
TYROSINE_RING_HYDROGENS = ["HD1", "HD2", "HE1", "HE2", "HH"]


@pytest.mark.parametrize(
    "residue_id, cut, removed, hydrogens",
    [
        (2, "CA", ["CB", "HB2", "HB3"] + TYROSINE_RING + TYROSINE_RING_HYDROGENS, ["HA"]),
        (2, "CB", TYROSINE_RING + TYROSINE_RING_HYDROGENS, ["HB2", "HB3"]),
        (5, "CG", ["CD", "OE1", "OE2"], ["HG2", "HG3"]),
    ],
)
def test_a_truncated_side_chain_keeps_its_types(
    frame, residue_id, cut, removed, hydrogens
):
    complete = parameters(frame)
    edited = drop(frame, residue_id, removed)
    actual = parameters(edited)
    for name in [cut, *hydrogens]:
        np.testing.assert_array_equal(
            actual.a[row(edited, residue_id, name)],
            complete.a[row(frame, residue_id, name)],
        )


def test_a_type_the_table_lacks_takes_peng(frame):
    edited = frame[frame["residue_id"] != 3].reset_index(drop=True)
    xyz = ["x", "y", "z"]
    n = row(edited, 4, "N")
    position = edited.iloc[n][xyz].to_numpy(dtype=float)
    hydrogens = edited.iloc[[row(edited, 4, "HA")] * 2].assign(atom=["H2", "H3"])
    for k, neighbour in enumerate(("CA", "CD")):
        away = position - edited.iloc[row(edited, 4, neighbour)][xyz].to_numpy(float)
        hydrogens.iloc[k, [hydrogens.columns.get_loc(c) for c in xyz]] = (
            position + away / np.linalg.norm(away)
        )
    edited = pd.concat([edited, hydrogens], ignore_index=True)
    assert types(edited)[n] == "N(HHCC)"
    actual = parameters(edited)
    peng = PengScatteringFactorParameters(np.array([7]))
    np.testing.assert_array_equal(actual.a[n], peng.a[0])
    np.testing.assert_array_equal(actual.b[n], peng.b[0])


def test_every_atom_taking_peng_raises(frame):
    renamed = frame.assign(residue="UNK")
    with pytest.raises(ValueError, match="No atom has a type in the 'catalase' table"):
        parameters(renamed)


def test_a_break_n_with_its_peptide_hydrogen_types_as_backbone(frame):
    edited = frame[frame["residue_id"] != 5].reset_index(drop=True)
    assert types(edited)[row(edited, 6, "N")] == "N(HCC)"
    parameters(edited)


def test_a_break_proline_n_without_hydrogen_types_as_backbone(frame):
    edited = frame[frame["residue_id"] != 3].reset_index(drop=True)
    assert types(edited)[row(edited, 4, "N")] == "N(CCC)"
    parameters(edited)


def test_a_break_n_with_two_hydrogens_is_not_a_cut(frame):
    edited = frame[frame["residue_id"] != 5].reset_index(drop=True)
    xyz = ["x", "y", "z"]
    n = edited.iloc[row(edited, 6, "N")][xyz].to_numpy(dtype=float)
    ca = edited.iloc[row(edited, 6, "CA")][xyz].to_numpy(dtype=float)
    hydrogen = edited.iloc[[row(edited, 6, "H")]].copy()
    hydrogen[xyz] = [n + (n - ca) / np.linalg.norm(n - ca)]
    edited = pd.concat([edited, hydrogen], ignore_index=True)
    assert types(edited)[row(edited, 6, "N")] == "N(HHC)"


def test_a_break_c_without_oxt_types_as_backbone(frame):
    edited = drop(frame, 10, ["OXT"])
    assert types(edited)[row(edited, 10, "C")] == "C(CNO)"
    assert types(frame)[row(frame, 10, "C")] == "C(COO)"


def test_lengths_must_match(frame):
    _, atomic_numbers, atom_chemistry = mmdf_to_atoms(frame, loads_chemistry=True)
    with pytest.raises(
        ValueError, match="`atomic_numbers` has 137 atoms, but `atom_chemistry` has 138"
    ):
        EmpiricalScatteringFactorParameters(atomic_numbers[:-1], atom_chemistry)


@pytest.mark.parametrize(
    "selection_string",
    [
        "resSeq 3 to 8",
        "not element H",
        "not name OG1",
        "not (name CB or name HB or name HB2 or name HB3 or name OG1 or name HG1)",
    ],
)
def test_selection_keeps_full_model_parameters(frame, selection_string):
    complete = parameters(frame)
    selected = parameters(frame, selection_string=selection_string)
    topology = mmdf_to_topology(frame)
    rows = topology.select(selection_string)
    np.testing.assert_array_equal(selected.a, complete.a[rows])
    np.testing.assert_array_equal(selected.b, complete.b[rows])


def test_fallback_to_peng_for_missing_type():
    atomic_numbers = np.array([7, 6, 6, 1, 1])
    atom_chemistry = dict(
        residue_names=np.array(["PRO"] * 5),
        atom_names=np.array(["N", "CA", "CD", "H2", "H3"]),
        bonds=np.array([[0, 1], [0, 2], [0, 3], [0, 4]]),
        missing_bonds=np.zeros((0, 2), dtype=int),
        unselected_bonds=np.zeros((0, 2), dtype=int),
    )
    assert _atom_types(atomic_numbers, atom_chemistry)[0] == "N(HHCC)"
    catalase = EmpiricalScatteringFactorParameters(atomic_numbers, atom_chemistry)
    peng = PengScatteringFactorParameters(np.array([7]))
    np.testing.assert_array_equal(catalase.a[0], peng.a[0])
    np.testing.assert_array_equal(catalase.b[0], peng.b[0])


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


@pytest.mark.parametrize("table", ["emdb", "peng"])
def test_table_keyword_takes_only_catalase(frame, table):
    with pytest.raises(ValueError, match="takes `table` equal to 'catalase'"):
        parameters(frame, table)


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
