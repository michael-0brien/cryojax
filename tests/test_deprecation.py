import importlib
import re

import cryojax
import cryojax.simulator as cxs
import pytest
from packaging.version import parse as parse_version


def test_future_deprecated(sample_pdb_path):
    match = re.match(r"(\d+\.\d+(?:\.\d+)?)", cryojax.__version__)
    assert match, f"Could not parse current cryojax version {cryojax.__version__!r}"
    current_version = parse_version(match.group(1))

    def should_be_removed(_record):
        msg = str(_record[0].message)
        match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", msg)
        assert match, f"Could not parse removal version from warning message: {msg}"
        removal_version = parse_version(match.group(1))
        return current_version >= removal_version

    # Old CTF aliases
    with pytest.warns(DeprecationWarning) as record:
        obj = cxs.AberratedAstigmaticCTF
        assert obj is cxs.AstigmaticCTF
        assert not should_be_removed(record)

    with pytest.warns(DeprecationWarning) as record:
        obj = cxs.CTF
        assert obj is cxs.AstigmaticCTF
        assert not should_be_removed(record)

    # Old volume-related interfaces
    with pytest.warns(DeprecationWarning) as record:
        obj = cxs.PengScatteringFactorParameters
        assert obj is cryojax.constants.PengScatteringFactorParameters
        assert not should_be_removed(record)

    with pytest.warns(DeprecationWarning) as record:
        obj = cxs.PengAtomicVolume
        assert obj is cryojax.simulator.GaussianMixtureVolume
        assert not should_be_removed(record)

    with pytest.warns(DeprecationWarning) as record:
        atom_pos, _, _ = cryojax.io.read_atoms_from_pdb(
            sample_pdb_path,
            loads_b_factors=True,
        )
        assert not should_be_removed(record)

    with pytest.warns(DeprecationWarning) as record:
        volume = cxs.GaussianMixtureVolume(atom_pos, amplitudes=1.0, variances=1.0)
        _ = volume.to_real_voxel_grid((32, 32, 32), 2.0)
        assert not should_be_removed(record)


def test_deprecated():
    DEPRECATED = [
        "cryojax.simulator.DiscreteStructuralEnsemble",
        "cryojax.simulator.CorrelatedGaussianNoiseModel",
        "cryojax.simulator.UncorrelatedGaussianNoiseModel",
    ]

    # Deprecated features
    for path in DEPRECATED:
        mod_path, _, attr = path.rpartition(".")
        module = importlib.import_module(mod_path)
        with pytest.raises(ValueError):
            _ = getattr(module, attr)
