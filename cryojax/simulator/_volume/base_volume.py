import abc
from typing import Generic, TypeVar
from typing_extensions import Self, override

import equinox as eqx
from jaxtyping import Array, Complex, Float, Inexact, PRNGKeyArray

from ...jax_util import NDArrayLike
from .._image_config import AbstractImageConfig
from .._pose import AbstractPose


VolRep = TypeVar("VolRep", bound="AbstractVolumeRepresentation")
T = TypeVar("T")


ProjectionOrEwaldSphereArray = (
    Complex[
        Array,
        "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
    ]
    | Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
    | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
)
ProjectionArray = (
    Complex[
        Array,
        "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
    ]
    | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
)
EwaldSphereArray = (
    Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
    | Float[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]
)

VoxelArray = (
    Inexact[Array, "{self.shape[0]} {self.shape[1]} {self.shape[2]}"]
    | Complex[Array, "{self.shape[0]} {self.shape[1]} {self.shape[2]}//2+1"]
)


class AbstractVolumeParametrization(eqx.Module, strict=True):
    """Abstract interface for a parametrization of a volume. Specifically,
    the cryo-EM image formation process typically starts with a *scattering potential*.
    "Volumes" and "scattering potentials" in cryoJAX are synonymous.

    !!! info
        In, `cryojax`, potentials should be built in units of *inverse length squared*,
        $[L]^{-2}$. This rescaled potential is defined to be

        $$U(\\mathbf{r}) = \\frac{m_0 e}{2 \\pi \\hbar^2} V(\\mathbf{r}),$$

        where $V$ is the electrostatic potential energy, $\\mathbf{r}$ is a positional
        coordinate, $m_0$ is the electron rest mass, and $e$ is the electron charge.

        For a single atom, this rescaled potential has the advantage that under usual
        scattering approximations (i.e. the first-born approximation), the
        fourier transform of this quantity is closely related to tabulated electron scattering
        factors. In particular, for a single atom with scattering factor $f^{(e)}(\\mathbf{q})$
        and scattering vector $\\mathbf{q}$, its rescaled potential is equal to

        $$U(\\mathbf{r}) = \\mathcal{F}^{-1}[f^{(e)}(\\boldsymbol{\\xi} / 2)](\\mathbf{r}),$$

        where $\\boldsymbol{\\xi} = 2 \\mathbf{q}$ is the wave vector coordinate and
        $\\mathcal{F}^{-1}$ is the inverse fourier transform operator in the convention

        $$\\mathcal{F}[f](\\boldsymbol{\\xi}) = \\int d^3\\mathbf{r} \\ \\exp(2\\pi i \\boldsymbol{\\xi}\\cdot\\mathbf{r}) f(\\mathbf{r}).$$

        The rescaled potential $U$ gives the following time-independent schrodinger equation
        for the scattering problem,

        $$(\\nabla^2 + k^2) \\psi(\\mathbf{r}) = - 4 \\pi U(\\mathbf{r}) \\psi(\\mathbf{r}),$$

        where $k$ is the incident wavenumber of the electron beam.

        **References**:

        - For the definition of the rescaled potential, see
        Chapter 69, Page 2003, Equation 69.6 from *Hawkes, Peter W., and Erwin Kasper.
        Principles of Electron Optics, Volume 4: Advanced Wave Optics. Academic Press,
        2022.*
        - To work out the correspondence between the rescaled potential and the electron
        scattering factors, see the supplementary information from *Vulović, Miloš, et al.
        "Image formation modeling in cryo-electron microscopy." Journal of structural
        biology 183.1 (2013): 19-32.*
    """  # noqa: E501

    @abc.abstractmethod
    def to_representation(
        self, rng_key: PRNGKeyArray | None = None
    ) -> "AbstractVolumeRepresentation":
        """Core interface for computing the representation of
        the volume.

        **Arguments:**

        - `rng_key`:
            An optional RNG key for including noise / stochastic
            elements to volume simulation.
        """
        raise NotImplementedError


class AbstractVolumeRepresentation(AbstractVolumeParametrization, strict=True):
    """Abstract interface for the representation of a volume, such
    as atomic coordinates, voxels, or a neural network.

    Volume representations contain information of coordinates and may be
    passed to `AbstractVolumeIntegrator` classes for imaging.
    """

    @abc.abstractmethod
    def rotate_to_pose(self, pose: AbstractPose, inverse: bool = False) -> Self:
        """Rotate the coordinate system of the volume."""
        raise NotImplementedError

    @override
    def to_representation(self, rng_key: PRNGKeyArray | None = None) -> Self:
        """Since this class is itself an
        `AbstractVolumeRepresentation`, this function maps to the identity.

        **Arguments:**

        - `rng_key`:
            Not used in this implementation.
        """
        del rng_key
        return self


class AbstractAtomVolume(AbstractVolumeRepresentation, strict=True):
    """Abstract interface for a volume represented as a point-cloud."""

    @abc.abstractmethod
    def translate_to_pose(self, pose: AbstractPose) -> Self:
        raise NotImplementedError


class AbstractVoxelVolume(AbstractVolumeRepresentation, strict=True):
    """Abstract interface for a volume represented with voxels.

    !!! info

        If you are using a `volume` in a voxel representation
        pass, the voxel size *must* be passed as the
        `pixel_size` argument, e.g.

        ```python
        import cryojax.simulator as cxs
        from cryojax.io import read_array_from_mrc

        real_voxel_grid, voxel_size = read_array_from_mrc("example.mrc")
        volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)
        ...
        config = cxs.BasicImageConfig(shape, pixel_size=voxel_size, ...)
        ```

        If this is not done, the resulting
        image will be incorrect and *not* rescaled to the specified
        to the different pixel size.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the voxel array."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_real_voxel_grid(
        cls, real_voxel_grid: Float[NDArrayLike, "dim dim dim"]
    ) -> Self:
        """Load an `AbstractVoxelVolume` from a 3D grid in
        real-space.
        """
        raise NotImplementedError


class AbstractVolumeIntegrator(eqx.Module, Generic[VolRep], strict=True):
    """Base class for a method of integrating a volume onto
    the exit plane.
    """

    outputs_ewald_sphere: eqx.AbstractClassVar[bool]

    @abc.abstractmethod
    def integrate(
        self,
        volume_representation: VolRep,
        image_config: AbstractImageConfig,
        outputs_real_space: bool = False,
    ) -> ProjectionOrEwaldSphereArray:
        raise NotImplementedError


class AbstractVolumeRenderFn(eqx.Module, Generic[VolRep], strict=True):
    """Base class for rendering a volume onto voxels."""

    shape: eqx.AbstractVar[tuple[int, int, int]]
    voxel_size: eqx.AbstractVar[Float[Array, ""]]

    @abc.abstractmethod
    def __call__(
        self,
        volume_representation: VolRep,
        *,
        outputs_real_space: bool = True,
        outputs_rfft: bool = False,
        fftshifted: bool = False,
    ) -> Array:
        raise NotImplementedError
