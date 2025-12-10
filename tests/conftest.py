import os

import jax
import pytest


# jax.config.update("jax_numpy_dtype_promotion", "strict")
# jax.config.update("jax_numpy_rank_promotion", "raise")
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sample_mrc_path():
    return os.path.join(os.path.dirname(__file__), "data", "3j9g_potential_ps4_4.mrc")


@pytest.fixture
def sample_pdb_path():
    return os.path.join(os.path.dirname(__file__), "data", "1uao.pdb")


@pytest.fixture
def sample_pdb_path_assembly():
    return os.path.join(os.path.dirname(__file__), "data", "1uao_assembly.pdb")


@pytest.fixture
def sample_cif_path():
    return os.path.join(os.path.dirname(__file__), "data", "8zpm.cif")


@pytest.fixture
def sample_waterbox_pdb():
    return os.path.join(os.path.dirname(__file__), "data", "relaxed_small_box_tip3p.pdb")


@pytest.fixture
def peng_parameters_path():
    return os.path.join(os.path.dirname(__file__), "data", "peng1996_params.npy")


@pytest.fixture
def lobato_parameters_path():
    return os.path.join(os.path.dirname(__file__), "data", "lobato2014_params.npy")
