from collections import Counter

import mdtraj
import mmdf
import numpy as np
import pandas as pd
import pytest
from cryojax.io import mmdf_to_atoms
from cryojax.io._chemistry import _AMINO_ACID_BONDS


def hydrogens_on(bonds):
    return dict(
        Counter(a if b.startswith("H") else b for a, b in bonds if "H" in (a[0], b[0]))
    )


def mdtraj_heavy_bonds(residue):
    mdtraj.Topology().create_standard_bonds()
    return {
        frozenset(bond)
        for bond in mdtraj.core.topology.Topology._standardBonds[residue]
        if not any(name[0] in "-+H" or name == "OXT" for name in bond)
    }


def test_amino_acid_heavy_bonds_match_mdtraj_templates():
    assert len(_AMINO_ACID_BONDS) == 20
    for residue, bonds in _AMINO_ACID_BONDS.items():
        heavy = {frozenset(b) for b in bonds if not any(n.startswith("H") for n in b)}
        assert heavy == mdtraj_heavy_bonds(residue), residue


def test_amino_acid_hydrogen_counts():
    assert hydrogens_on(_AMINO_ACID_BONDS["SER"]) == {"N": 1, "CA": 1, "CB": 2, "OG": 1}
    assert hydrogens_on(_AMINO_ACID_BONDS["PRO"]) == {"CA": 1, "CB": 2, "CG": 2, "CD": 2}
    assert hydrogens_on(_AMINO_ACID_BONDS["HIS"])["ND1"] == 1
    assert hydrogens_on(_AMINO_ACID_BONDS["HIS"])["NE2"] == 1
    assert hydrogens_on(_AMINO_ACID_BONDS["LYS"])["NZ"] == 3
    assert "OD2" not in hydrogens_on(_AMINO_ACID_BONDS["ASP"])


@pytest.fixture
def frame(sample_pdb_path):
    return mmdf.read(sample_pdb_path)


def row(df, residue_id, atom):
    return int(np.flatnonzero((df["residue_id"] == residue_id) & (df["atom"] == atom))[0])


def drop(df, residue_id, atoms):
    keep = ~((df["residue_id"] == residue_id) & df["atom"].isin(atoms))
    return df[keep].reset_index(drop=True)


def new_atom(template, **fields):
    atom = template.copy()
    for name, value in fields.items():
        atom[name] = value
    return atom


def chemistry(df, **kwargs):
    return mmdf_to_atoms(df, loads_chemistry=True, **kwargs)[2]


@pytest.mark.parametrize("selection_string", ["all", "resSeq 3 to 8"])
def test_chemistry_keeps_rows(frame, selection_string):
    positions, atomic_numbers, _ = mmdf_to_atoms(
        frame, loads_chemistry=True, selection_string=selection_string
    )
    expected_positions, expected_numbers = mmdf_to_atoms(
        frame, selection_string=selection_string
    )
    np.testing.assert_array_equal(positions, expected_positions)
    np.testing.assert_array_equal(atomic_numbers, expected_numbers)


def test_bonds_match_mdtraj_on_1uao(frame, sample_pdb_path):
    bonds = chemistry(frame)["bonds"]
    assert bonds.shape == (141, 2)
    topology = mdtraj.load(sample_pdb_path).topology
    assert {frozenset(b) for b in bonds.tolist()} == {
        frozenset((a.index, b.index)) for a, b in topology.bonds
    }


def test_complete_model_has_no_missing_bonds(frame):
    assert chemistry(frame)["missing_bonds"].shape == (0, 2)


def test_missing_oxt(frame):
    edited = drop(frame, 10, ["OXT"])
    np.testing.assert_array_equal(
        chemistry(edited)["missing_bonds"], [[row(edited, 10, "C"), 8]]
    )


def test_missing_hydroxyl_hydrogen(frame):
    edited = drop(frame, 6, ["HG1"])
    np.testing.assert_array_equal(
        chemistry(edited)["missing_bonds"], [[row(edited, 6, "OG1"), 1]]
    )


def test_truncated_side_chain(frame):
    ring = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "HD1", "HD2", "HE1", "HE2"]
    edited = drop(frame, 2, ring + ["HH"])
    np.testing.assert_array_equal(
        chemistry(edited)["missing_bonds"], [[row(edited, 2, "CB"), 6]]
    )


def test_selection_cut_records_missing_peptide_bonds(frame):
    kept = frame[frame["residue_id"].between(3, 8)].reset_index(drop=True)
    np.testing.assert_array_equal(
        chemistry(frame, selection_string="resSeq 3 to 8")["missing_bonds"],
        [[row(kept, 3, "N"), 6], [row(kept, 8, "C"), 7]],
    )


def test_hydrogen_names_do_not_matter(frame):
    renamed = frame.copy()
    is_hydrogen = renamed["atomic_number"] == 1
    renamed.loc[is_hydrogen, "atom"] = [f"HX{i}" for i in range(is_hydrogen.sum())]
    expected, actual = chemistry(frame), chemistry(renamed)
    np.testing.assert_array_equal(actual["bonds"], expected["bonds"])
    np.testing.assert_array_equal(actual["missing_bonds"], expected["missing_bonds"])


def test_unknown_residues_have_no_bonds(frame):
    oxygen = frame[frame["atomic_number"] == 8].iloc[[0]]
    extra = pd.concat(
        [
            new_atom(oxygen, residue="GTP", residue_id=11, atom="PG", atomic_number=15),
            new_atom(oxygen, chain="B", residue="MG", atom="MG", atomic_number=12),
            new_atom(oxygen, chain="C", residue="HOH", atom="O", x=60.0),
            new_atom(oxygen, chain="C", residue="HOH", atom="H1", atomic_number=1),
        ]
    )
    edited = pd.concat([frame, extra], ignore_index=True)
    actual, expected = chemistry(edited), chemistry(frame)
    np.testing.assert_array_equal(actual["bonds"], expected["bonds"])
    np.testing.assert_array_equal(actual["missing_bonds"], expected["missing_bonds"])


def cysteine_pair(template, sg_distance):
    elements = {"N": 7, "C": 6, "O": 8, "S": 16}
    atoms = [
        new_atom(
            template,
            chain=chain,
            residue="CYS",
            residue_id=1,
            atom=name,
            element=name[0],
            atomic_number=elements[name[0]],
            x=origin + step * k,
            y=0.0,
            z=0.0,
        )
        for chain, origin, step in (("A", 0.0, 1.5), ("B", 15.0 + sg_distance, -1.5))
        for k, name in enumerate(["N", "CA", "C", "O", "CB", "SG"])
    ]
    return pd.concat(atoms, ignore_index=True)


def test_disulfide(frame):
    bonded = chemistry(cysteine_pair(frame.iloc[[0]], 2.04))
    assert [5, 11] in bonded["bonds"].tolist()
    assert [5, 1] not in bonded["missing_bonds"].tolist()
    apart = chemistry(cysteine_pair(frame.iloc[[0]], 4.0))
    assert [5, 11] not in apart["bonds"].tolist()
    assert [5, 1] in apart["missing_bonds"].tolist()
    assert [11, 1] in apart["missing_bonds"].tolist()


def test_repeated_atom_raises(frame):
    duplicated = pd.concat([frame, frame.iloc[[row(frame, 2, "CB")]]], ignore_index=True)
    with pytest.raises(
        ValueError,
        match="an alternate conformation, an insertion code or a repeated chain",
    ):
        chemistry(duplicated)


def test_residues_sharing_a_number(frame):
    renumbered = frame.copy()
    renumbered.loc[renumbered["residue_id"] == 7, "residue_id"] = 6
    expected, actual = chemistry(frame), chemistry(renumbered)
    np.testing.assert_array_equal(actual["bonds"], expected["bonds"])
    np.testing.assert_array_equal(actual["missing_bonds"], expected["missing_bonds"])


def test_unexpected_hydrogen_raises(frame):
    xyz = ["x", "y", "z"]
    od2 = frame.iloc[row(frame, 3, "OD2")][xyz].to_numpy(dtype=float)
    cg = frame.iloc[row(frame, 3, "CG")][xyz].to_numpy(dtype=float)
    x, y, z = od2 + (od2 - cg) / np.linalg.norm(od2 - cg)
    hydrogen = new_atom(frame.iloc[[row(frame, 3, "H")]], atom="HD2", x=x, y=y, z=z)
    with pytest.raises(
        ValueError, match="too many hydrogens on atom OD2 .*: 1, but the CCP4 .* has 0"
    ):
        chemistry(pd.concat([frame, hydrogen], ignore_index=True))


def test_multi_model(sample_pdb_path, sample_pdb_path_assembly):
    single = chemistry(mmdf.read(sample_pdb_path))
    assembly = mmdf.read(sample_pdb_path_assembly)
    stacked = chemistry(assembly, stack_models=True)
    assert stacked["atom_names"].shape == (138,)
    np.testing.assert_array_equal(stacked["bonds"], single["bonds"])
    flat = chemistry(assembly)
    assert flat["atom_names"].shape == (1380,)
    assert flat["bonds"].shape == (1410, 2)
    assert np.all(flat["bonds"][:, 0] // 138 == flat["bonds"][:, 1] // 138)


def test_modified_residue_stays_in_chain(frame):
    modified = frame.copy()
    modified.loc[modified["residue_id"] == 6, "residue"] = "TPO"
    atom_chemistry = chemistry(modified)
    assert [row(frame, 5, "C"), row(frame, 6, "N")] in atom_chemistry["bonds"].tolist()
    assert [row(frame, 6, "C"), row(frame, 7, "N")] in atom_chemistry["bonds"].tolist()
    assert atom_chemistry["missing_bonds"].shape == (0, 2)
