from typing import Any
from typing_extensions import Self, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ....constants import convert_variance_to_b_factor
from ....jax_util import NDArrayLike, error_if_not_positive
from ..._pose import AbstractPose
from ..common_functions import gaussians_to_real_voxels
from .base_volume import AbstractPointCloudVolume


class GaussianMixtureVolume(AbstractPointCloudVolume, strict=True):
    r"""A representation of a volume as a mixture of
    gaussians, with multiple gaussians used per position.

    The convention of allowing multiple gaussians per position
    follows "Robust Parameterization of Elastic and Absorptive
    Electron Atomic Scattering Factors" by Peng et al. (1996), where
    `amplitudes` follows $a_i$ and $b_i$ follows `variances`
    (multiplied by $8\pi^2$ to convert to a variance).
    """

    positions: Float[Array, "n_positions 3"]
    amplitudes: Float[Array, "n_positions n_gaussians"]
    variances: Float[Array, " n_positions n_gaussians"]

    def __init__(
        self,
        positions: Float[NDArrayLike, "n_gaussians 3"],
        amplitudes: Float[NDArrayLike, "n_positions n_gaussians"],
        variances: Float[NDArrayLike, "n_positions n_gaussians"],
    ):
        """**Arguments:**

        - `positions`:
            The coordinates of the gaussians in units of angstroms.
        - `amplitudes`:
            The amplitude for each gaussian.
            To simulate in physical units of a scattering potential,
            this should have units of angstroms.
        - `variances`:
            The variance for each gaussian. This has units of angstroms
            squared.
        """
        self.positions = jnp.asarray(positions, dtype=float)
        self.amplitudes = jnp.asarray(amplitudes, dtype=float)
        self.variances = error_if_not_positive(jnp.asarray(variances, dtype=float))

    @override
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Return a new potential with rotated `positions`."""
        return eqx.tree_at(
            lambda d: d.positions,
            self,
            pose.rotate_coordinates(self.positions, inverse=inverse),
        )

    @override
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        """Return a new potential with rotated `positions`."""
        offset_in_angstroms = pose.offset_in_angstroms
        if pose.offset_z_in_angstroms is None:
            offset_in_angstroms = jnp.concatenate(
                (offset_in_angstroms, jnp.atleast_1d(0.0))
            )
        return eqx.tree_at(
            lambda d: d.positions, self, self.positions + offset_in_angstroms
        )

    def to_real_voxel_grid(
        self,
        shape: tuple[int, int, int],
        voxel_size: Float[NDArrayLike, ""] | float,
        *,
        batch_options: dict[str, Any] = {},
    ) -> Float[Array, "{shape[0]} {shape[1]} {shape[2]}"]:
        """Return a voxel grid of the potential in real space.

        **Arguments:**

        - `shape`: The shape of the resulting voxel grid.
        - `voxel_size`: The voxel size of the resulting voxel grid.
        - `batch_options`:
            Advanced options for rendering. This is a dictionary
            with the following keys:
            - "batch_size":
                The number of z-planes to evaluate in parallel with
                `jax.vmap`. By default, `1`.
            - "n_batches":
                The number of iterations used to evaluate the volume,
                where the iteration is taken over groups of gaussians.
                This is useful if `batch_size = 1`
                and GPU memory is exhausted. By default, `1`.

        **Returns:**

        A voxel grid of `shape` and voxel size `voxel_size`.
        """  # noqa: E501
        return gaussians_to_real_voxels(
            shape,
            jnp.asarray(voxel_size, dtype=float),
            self.positions,
            self.amplitudes,
            convert_variance_to_b_factor(self.variances),
            **batch_options,
        )
