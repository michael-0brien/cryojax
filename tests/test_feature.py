import math

from cryojax.feature import new_feature


def test_new_feature():
    assert math.isnan(new_feature())
