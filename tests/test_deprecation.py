import importlib
import re

import cryojax
import cryojax.simulator as cxs
import pytest
from packaging.version import parse as parse_version


def test_future_deprecated():
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


def test_deprecated():
    DEPRECATED = ["cryojax.simulator.DiscreteStructuralEnsemble"]

    # Deprecated features
    for path in DEPRECATED:
        mod_path, _, attr = path.rpartition(".")
        module = importlib.import_module(mod_path)
        with pytest.raises(ValueError):
            _ = getattr(module, attr)
