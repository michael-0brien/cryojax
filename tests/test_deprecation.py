import cryojax.simulator as cxs
import pytest


def test_deprecated():
    # Old CTF aliases
    with pytest.warns(DeprecationWarning):
        obj = cxs.AberratedAstigmaticCTF
        assert obj is cxs.AstigmaticCTF

    with pytest.warns(DeprecationWarning):
        obj = cxs.CTF
        assert obj is cxs.AstigmaticCTF
