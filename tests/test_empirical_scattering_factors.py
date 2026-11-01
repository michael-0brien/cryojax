from cryojax.constants._empirical_tables import (
    _CATALASE_TABLE,
    _EMDB_TABLE,
)


def test_table_sizes():
    assert len(_CATALASE_TABLE) == 42
    assert len(_EMDB_TABLE) == 96
    assert set(_CATALASE_TABLE) <= set(_EMDB_TABLE)
    for table in (_CATALASE_TABLE, _EMDB_TABLE):
        assert all(len(a) == len(b) == 5 for a, b in table.values())


def test_table_values():
    assert _CATALASE_TABLE["H(C)"] == (
        (-0.2309, 0.503, 0.5916, -0.9023, -0.4652),
        (0.6683, 16.7729, 9.7436, 49.7746, 87.8615),
    )
    assert _EMDB_TABLE["H(C)"] == (
        (-0.1529, 4.135, -4.9181, 1.4351, -1.5408),
        (0.076, 24.7071, 34.3897, 49.3086, 87.991),
    )
