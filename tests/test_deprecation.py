import cryojax as cx
import cryojax.simulator as cxs
import pytest


def test_future_deprecated(sample_pdb_path):
    """Template for testing FutureWarning deprecations scheduled for a future release.

    Copy and adapt the pattern below when adding new deprecations:

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
            <trigger deprecated usage here>
            assert not should_be_removed(record)
    """
    pass


def test_deprecated():
    # Names removed from cryojax.simulator
    REMOVED_FROM_SIMULATOR = [
        "AberratedAstigmaticCTF",
        "CTF",
        "NufftProjection",
        "PengScatteringFactorParameters",
        "PengAtomicVolume",
        "UncorrelatedGaussianNoiseModel",
        "CorrelatedGaussianNoiseModel",
        "DiscreteStructuralEnsemble",
    ]
    for name in REMOVED_FROM_SIMULATOR:
        with pytest.raises(AttributeError):
            _ = getattr(cxs, name)

    # Names removed from cryojax.ndimage
    REMOVED_FROM_NDIMAGE = [
        "downsample_with_fourier_cropping",
        "downsample_to_shape_with_fourier_cropping",
        "normalize_image",
        "operators",
        "transforms",
    ]
    for name in REMOVED_FROM_NDIMAGE:
        with pytest.raises(AttributeError):
            _ = getattr(cx.ndimage, name)

    # Submodules removed from top-level cryojax
    with pytest.raises(ImportError):
        _ = cx.coordinates  # type: ignore
    with pytest.raises(ImportError):
        _ = cx.dataset  # type: ignore

    # Grid search API removed from cryojax.jax_util
    REMOVED_FROM_JAX_UTIL = [
        "run_grid_search",
        "AbstractGridSearchMethod",
        "MinimumSearchMethod",
        "tree_grid_shape",
        "tree_grid_take",
        "tree_grid_unravel_index",
    ]
    for name in REMOVED_FROM_JAX_UTIL:
        with pytest.raises(AttributeError):
            _ = getattr(cx.jax_util, name)
