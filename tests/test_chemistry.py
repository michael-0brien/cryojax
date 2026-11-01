from collections import Counter

import mdtraj
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
