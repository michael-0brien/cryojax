import typing
from collections import Counter
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from jaxtyping import Int, Shaped
from scipy.spatial import KDTree


if hasattr(typing, "GENERATING_DOCUMENTATION"):
    _AtomChemistry = dict[str, Any]  # pyright: ignore[reportAssignmentType]
else:

    class _AtomChemistry(TypedDict):
        residue_names: Shaped[np.ndarray, " n_atoms"]
        atom_names: Shaped[np.ndarray, " n_atoms"]
        bonds: Int[np.ndarray, "n_bonds 2"]
        missing_bonds: Int[np.ndarray, "n_missing 2"]


def _read_chemistry(
    df: pd.DataFrame, residue_names: np.ndarray, atom_names: np.ndarray
) -> _AtomChemistry:
    atomic_numbers = df["atomic_number"].to_numpy()
    positions = df[["x", "y", "z"]].to_numpy(dtype=float)
    residues: dict[tuple, list[int]] = {}
    keys = zip(df["model"], df["chain"], df["residue_id"], residue_names)
    for row, key in enumerate(keys):
        residues.setdefault(key, []).append(row)
    amino_acids = [
        (key, rows) for key, rows in residues.items() if key[3] in _AMINO_ACID_BONDS
    ]
    heavy_atoms = [
        _heavy_atoms(key, rows, atom_names, atomic_numbers) for key, rows in amino_acids
    ]
    models, chains = df["model"].to_numpy(), df["chain"].to_numpy()
    residue_of = np.empty(len(df), dtype=int)
    for k, rows in enumerate(residues.values()):
        residue_of[rows] = k
    # A residue's C bonds the N of another residue of its chain.
    carbons = np.flatnonzero((atom_names == "C") & (atomic_numbers == 6))
    nitrogens = np.flatnonzero((atom_names == "N") & (atomic_numbers == 7))
    close = KDTree(positions[carbons]).sparse_distance_matrix(
        KDTree(positions[nitrogens]), _MAX_PEPTIDE_BOND, output_type="ndarray"
    )
    c, n = carbons[close["i"]], nitrogens[close["j"]]
    is_peptide = (
        (residue_of[c] != residue_of[n])
        & (models[c] == models[n])
        & (chains[c] == chains[n])
    )
    peptides = np.stack([c[is_peptide], n[is_peptide]], axis=-1)
    sulfurs = np.array(
        [
            atoms["SG"]
            for (key, _), atoms in zip(amino_acids, heavy_atoms)
            if key[3] == "CYS" and "SG" in atoms
        ],
        dtype=int,
    )
    disulfides = sulfurs[
        KDTree(positions[sulfurs]).query_pairs(_MAX_DISULFIDE_BOND, output_type="ndarray")
    ]
    disulfides = disulfides[models[disulfides[:, 0]] == models[disulfides[:, 1]]]
    bonds, missing_bonds = peptides.tolist() + disulfides.tolist(), []
    bonded_carbons = set(peptides[:, 0].tolist())
    bonded_nitrogens = set(peptides[:, 1].tolist())
    bridged = set(disulfides.ravel().tolist())
    for (key, rows), atoms in zip(amino_acids, heavy_atoms):
        residue_bonds = list(_AMINO_ACID_BONDS[key[3]])
        if atoms.get("C") not in bonded_carbons:
            residue_bonds.append(("C", "OXT"))
        expected_hydrogens = Counter(
            a if b.startswith("H") else b
            for a, b in residue_bonds
            if a.startswith("H") or b.startswith("H")
        )
        if atoms.get("N") not in bonded_nitrogens:
            expected_hydrogens["N"] += 2
        if atoms.get("SG") in bridged:
            expected_hydrogens["SG"] -= 1
        for a, b in residue_bonds:
            if a.startswith("H") or b.startswith("H"):
                continue
            row_a, row_b = atoms.get(a), atoms.get(b)
            if row_a is not None and row_b is not None:
                bonds.append((row_a, row_b))
            elif row_a is not None:
                missing_bonds.append((row_a, _ATOMIC_NUMBERS[b[0]]))
            elif row_b is not None:
                missing_bonds.append((row_b, _ATOMIC_NUMBERS[a[0]]))
        hydrogens = [row for row in rows if atomic_numbers[row] == 1]
        found = Counter()
        if hydrogens:
            if not atoms:
                raise ValueError(
                    f"Found hydrogens but no heavy atoms in {_describe(key)}."
                )
            heavy_rows = np.array(list(atoms.values()))
            distances = np.linalg.norm(
                positions[hydrogens, None] - positions[None, heavy_rows], axis=-1
            )
            parents = heavy_rows[np.argmin(distances, axis=1)]
            bonds.extend(zip(parents.tolist(), hydrogens))
            found = Counter(parents.tolist())
        for name, row in atoms.items():
            if found[row] > expected_hydrogens[name]:
                raise ValueError(
                    f"Found too many hydrogens on atom {name} of {_describe(key)}: "
                    f"{found[row]}, but the CCP4 Monomer Library's protonation, with "
                    f"charged termini at chain ends and breaks, has "
                    f"{expected_hydrogens[name]}."
                )
            missing_bonds.extend([(row, 1)] * (expected_hydrogens[name] - found[row]))
    return _AtomChemistry(
        residue_names=residue_names,
        atom_names=atom_names,
        bonds=_as_pairs(bonds, sorts_each=True),
        missing_bonds=_as_pairs(missing_bonds, sorts_each=False),
    )


def _heavy_atoms(key, rows, atom_names, atomic_numbers):
    known = {name for bond in _AMINO_ACID_BONDS[key[3]] for name in bond} | {"OXT"}
    atoms = {}
    for row in rows:
        if atomic_numbers[row] == 1:
            continue
        name = atom_names[row]
        if name not in known:
            raise ValueError(f"Found an unknown atom {name} in {_describe(key)}.")
        if name in atoms:
            raise ValueError(
                f"Found atom {name} twice in {_describe(key)}. `mmdf` keeps no "
                "alternate-location labels or insertion codes, so this is an alternate "
                "conformation, an insertion code or a repeated chain ID; keep one copy "
                "of each atom per residue."
            )
        atoms[name] = row
    return atoms


def _select_chemistry(atom_chemistry, selected, atomic_numbers):
    index = np.full(len(atomic_numbers), -1)
    index[selected] = np.arange(len(selected))
    bonds, missing_bonds = atom_chemistry["bonds"], atom_chemistry["missing_bonds"]
    kept = index[bonds] >= 0
    both, first = kept.all(axis=1), kept[:, 0] & ~kept[:, 1]
    second = kept[:, 1] & ~kept[:, 0]
    cut = np.concatenate(
        [
            np.stack([bonds[first, 0], atomic_numbers[bonds[first, 1]]], axis=-1),
            np.stack([bonds[second, 1], atomic_numbers[bonds[second, 0]]], axis=-1),
            missing_bonds[index[missing_bonds[:, 0]] >= 0],
        ]
    )
    cut[:, 0] = index[cut[:, 0]]
    return _AtomChemistry(
        residue_names=atom_chemistry["residue_names"][selected],
        atom_names=atom_chemistry["atom_names"][selected],
        bonds=_as_pairs(index[bonds[both]], sorts_each=True),
        missing_bonds=_as_pairs(cut, sorts_each=False),
    )


def _as_pairs(pairs, sorts_each):
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if sorts_each:
        pairs = np.sort(pairs, axis=1)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _describe(key):
    model, chain, residue_id, residue = key
    return f"residue {residue} {residue_id} of chain {chain} (model {model})"


_ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
_MAX_PEPTIDE_BOND = 1.5 * 1.341
_MAX_DISULFIDE_BOND = 1.5 * 2.04

# Mid-chain bonds of the standard amino acids, from the CCP4 Monomer Library.
_AMINO_ACID_BONDS: dict[str, tuple[tuple[str, str], ...]] = {
    "ALA": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "HB1"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "ARG": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "HD2"),
        ("CD", "HD3"),
        ("CD", "NE"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
        ("N", "CA"),
        ("N", "H"),
        ("NE", "CZ"),
        ("NE", "HE"),
        ("NH1", "HH11"),
        ("NH1", "HH12"),
        ("NH2", "HH21"),
        ("NH2", "HH22"),
    ),
    "ASN": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CG", "ND2"),
        ("CG", "OD1"),
        ("N", "CA"),
        ("N", "H"),
        ("ND2", "HD21"),
        ("ND2", "HD22"),
    ),
    "ASP": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CG", "OD1"),
        ("CG", "OD2"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "CYS": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CB", "SG"),
        ("N", "CA"),
        ("N", "H"),
        ("SG", "HG"),
    ),
    "GLN": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "NE2"),
        ("CD", "OE1"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("N", "CA"),
        ("N", "H"),
        ("NE2", "HE21"),
        ("NE2", "HE22"),
    ),
    "GLU": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "OE1"),
        ("CD", "OE2"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "GLY": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "HA2"),
        ("CA", "HA3"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "HIS": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD2", "HD2"),
        ("CD2", "NE2"),
        ("CE1", "HE1"),
        ("CE1", "NE2"),
        ("CG", "CD2"),
        ("CG", "ND1"),
        ("N", "CA"),
        ("N", "H"),
        ("ND1", "CE1"),
        ("ND1", "HD1"),
        ("NE2", "HE2"),
    ),
    "ILE": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG1"),
        ("CB", "CG2"),
        ("CB", "HB"),
        ("CD1", "HD11"),
        ("CD1", "HD12"),
        ("CD1", "HD13"),
        ("CG1", "CD1"),
        ("CG1", "HG12"),
        ("CG1", "HG13"),
        ("CG2", "HG21"),
        ("CG2", "HG22"),
        ("CG2", "HG23"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "LEU": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD1", "HD11"),
        ("CD1", "HD12"),
        ("CD1", "HD13"),
        ("CD2", "HD21"),
        ("CD2", "HD22"),
        ("CD2", "HD23"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CG", "HG"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "LYS": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "CE"),
        ("CD", "HD2"),
        ("CD", "HD3"),
        ("CE", "HE2"),
        ("CE", "HE3"),
        ("CE", "NZ"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("N", "CA"),
        ("N", "H"),
        ("NZ", "HZ1"),
        ("NZ", "HZ2"),
        ("NZ", "HZ3"),
    ),
    "MET": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CE", "HE1"),
        ("CE", "HE2"),
        ("CE", "HE3"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("CG", "SD"),
        ("N", "CA"),
        ("N", "H"),
        ("SD", "CE"),
    ),
    "PHE": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD1", "CE1"),
        ("CD1", "HD1"),
        ("CD2", "CE2"),
        ("CD2", "HD2"),
        ("CE1", "CZ"),
        ("CE1", "HE1"),
        ("CE2", "CZ"),
        ("CE2", "HE2"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CZ", "HZ"),
        ("N", "CA"),
        ("N", "H"),
    ),
    "PRO": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD", "HD2"),
        ("CD", "HD3"),
        ("CG", "CD"),
        ("CG", "HG2"),
        ("CG", "HG3"),
        ("N", "CA"),
        ("N", "CD"),
    ),
    "SER": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CB", "OG"),
        ("N", "CA"),
        ("N", "H"),
        ("OG", "HG"),
    ),
    "THR": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG2"),
        ("CB", "HB"),
        ("CB", "OG1"),
        ("CG2", "HG21"),
        ("CG2", "HG22"),
        ("CG2", "HG23"),
        ("N", "CA"),
        ("N", "H"),
        ("OG1", "HG1"),
    ),
    "TRP": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD1", "HD1"),
        ("CD1", "NE1"),
        ("CD2", "CE2"),
        ("CD2", "CE3"),
        ("CE2", "CZ2"),
        ("CE3", "CZ3"),
        ("CE3", "HE3"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CH2", "HH2"),
        ("CZ2", "CH2"),
        ("CZ2", "HZ2"),
        ("CZ3", "CH2"),
        ("CZ3", "HZ3"),
        ("N", "CA"),
        ("N", "H"),
        ("NE1", "CE2"),
        ("NE1", "HE1"),
    ),
    "TYR": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG"),
        ("CB", "HB2"),
        ("CB", "HB3"),
        ("CD1", "CE1"),
        ("CD1", "HD1"),
        ("CD2", "CE2"),
        ("CD2", "HD2"),
        ("CE1", "CZ"),
        ("CE1", "HE1"),
        ("CE2", "CZ"),
        ("CE2", "HE2"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CZ", "OH"),
        ("N", "CA"),
        ("N", "H"),
        ("OH", "HH"),
    ),
    "VAL": (
        ("C", "O"),
        ("CA", "C"),
        ("CA", "CB"),
        ("CA", "HA"),
        ("CB", "CG1"),
        ("CB", "CG2"),
        ("CB", "HB"),
        ("CG1", "HG11"),
        ("CG1", "HG12"),
        ("CG1", "HG13"),
        ("CG2", "HG21"),
        ("CG2", "HG22"),
        ("CG2", "HG23"),
        ("N", "CA"),
        ("N", "H"),
    ),
}
