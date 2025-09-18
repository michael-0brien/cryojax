import abc
from typing import Optional, TypeVar
from typing_extensions import Self, override

import equinox as eqx
from jaxtyping import PRNGKeyArray

from .._pose import AbstractPose


T = TypeVar("T")


#
# Core base class for parametrising a volume
#
class AbstractVolumeParametrisation(eqx.Module, strict=True):
    """Abstract interface for a parametrisation of a volume. Specifically,
    the cryo-EM image formation process typically starts with a *scattering potential*.
    "Volumes" and "scattering potentials" in cryoJAX are synonymous.

    !!! info
        In, `cryojax`, potentials should be built in units of *inverse length squared*,
        $[L]^{-2}$. This rescaled potential is defined to be

        $$U(\\mathbf{r}) = \\frac{2 m e}{\\hbar^2} V(\\mathbf{r}),$$

        where $V$ is the electrostatic potential energy, $\\mathbf{r}$ is a positional
        coordinate, $m$ is the electron mass, and $e$ is the electron charge.

        For a single atom, this rescaled potential has the advantage that under usual
        scattering approximations (i.e. the first-born approximation), the
        fourier transform of this quantity is closely related to tabulated electron scattering
        factors. In particular, for a single atom with scattering factor $f^{(e)}(\\mathbf{q})$
        and scattering vector $\\mathbf{q}$, its rescaled potential is equal to

        $$U(\\mathbf{r}) = 4 \\pi \\mathcal{F}^{-1}[f^{(e)}(\\boldsymbol{\\xi} / 2)](\\mathbf{r}),$$

        where $\\boldsymbol{\\xi} = 2 \\mathbf{q}$ is the wave vector coordinate and
        $\\mathcal{F}^{-1}$ is the inverse fourier transform operator in the convention

        $$\\mathcal{F}[f](\\boldsymbol{\\xi}) = \\int d^3\\mathbf{r} \\ \\exp(2\\pi i \\boldsymbol{\\xi}\\cdot\\mathbf{r}) f(\\mathbf{r}).$$

        The rescaled potential $U$ gives the following time-independent schrodinger equation
        for the scattering problem,

        $$(\\nabla^2 + k^2) \\psi(\\mathbf{r}) = - U(\\mathbf{r}) \\psi(\\mathbf{r}),$$

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
    def compute_volume_representation(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> "AbstractVolumeRepresentation":
        """Core interface for computing the representation of
        the volume.
        """
        raise NotImplementedError


#
# Interfaces that give core properties to volumes
#
class AbstractVolumeRepresentation(AbstractVolumeParametrisation, strict=True):
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
    def compute_volume_representation(
        self, rng_key: Optional[PRNGKeyArray] = None
    ) -> Self:
        """Since this class is itself an implementation of an
        `AbstractVolumeParametrisation`, this function maps to the identity.

        **Arguments:**

        - `rng_key`:
            Not used in this implementation, but optionally
            included for other implementations.
        """
        return self
