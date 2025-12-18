import importlib
import re

import cryojax as cx
import cryojax.simulator as cxs
import pytest
from packaging.version import parse as parse_version


def test_future_deprecated(sample_pdb_path):
    match = re.match(r"(\d+\.\d+(?:\.\d+)?)", cx.__version__)
    assert match, f"Could not parse current cryojax version {cx.__version__!r}"
    current_version = parse_version(match.group(1))

    def should_be_removed(_record):
        msg = str(_record[0].message)
        match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", msg)
        assert match, f"Could not parse removal version from warning message: {msg}"
        removal_version = parse_version(match.group(1))
        return current_version >= removal_version

    with pytest.warns(FutureWarning) as record:
        obj = cxs.AberratedAstigmaticCTF
        assert obj is cxs.AstigmaticCTF
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        obj = cxs.CTF
        assert obj is cxs.AstigmaticCTF
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        obj = cxs.CorrelatedGaussianNoiseModel
        assert obj is cxs.GaussianColoredNoiseModel
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        obj = cxs.UncorrelatedGaussianNoiseModel
        assert obj is cxs.GaussianWhiteNoiseModel
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        obj = cxs.NufftProjection
        assert obj is cxs.RealVoxelProjection
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        obj = cxs.PengScatteringFactorParameters
        assert obj is cx.constants.PengScatteringFactorParameters
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        obj = cxs.PengAtomicVolume
        assert obj is cxs.GaussianMixtureVolume
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        atom_pos, _, _ = cx.io.read_atoms_from_pdb(  # type: ignore
            sample_pdb_path,
            loads_b_factors=True,
        )
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        volume = cxs.GaussianMixtureVolume(atom_pos, amplitudes=1.0, variances=1.0)
        _ = volume.to_real_voxel_grid((32, 32, 32), 2.0)
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        _ = cxs.GaussianMixtureProjection(use_error_functions=True)  # type: ignore
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        func = cx.ndimage.downsample_with_fourier_cropping
        assert func is cx.ndimage.fourier_crop_downsample
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        func = cx.ndimage.downsample_to_shape_with_fourier_cropping
        assert func is cx.ndimage.fourier_crop_to_shape
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        func = cx.ndimage.normalize_image
        assert func is cx.ndimage.standardize_image
        assert not should_be_removed(record)

    # ndimage submodules
    with pytest.warns(FutureWarning) as record:
        from cryojax.coordinates import make_frequency_grid as make_1
        from cryojax.ndimage import make_frequency_grid as make_2

        assert make_1 is make_2
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        from cryojax.ndimage import AbstractFourierOperator, operators as op

        assert AbstractFourierOperator is op.AbstractFourierOperator
        assert not should_be_removed(record)

    with pytest.warns(FutureWarning) as record:
        from cryojax.ndimage import AbstractFilter, transforms as tf

        assert AbstractFilter is tf.AbstractFilter
        assert not should_be_removed(record)


def test_deprecated():
    DEPRECATED = [
        "cryojax.simulator.DiscreteStructuralEnsemble",
    ]

    # Deprecated features
    for path in DEPRECATED:
        mod_path, _, attr = path.rpartition(".")
        module = importlib.import_module(mod_path)
        with pytest.raises(ImportError):
            _ = getattr(module, attr)
