from abc import abstractmethod
from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ..._internal import error_if_negative, leaf_asarray
from ...jax_util import FloatLike, NDArrayLike
from .phase_shifts import (
    compute_amplitude_contrast_phase_shift,
    compute_even_aberration_phase_shifts,
    compute_odd_aberration_phase_shifts,
)


Parity = Literal["all", "even", "odd"]


class AbstractCTF(eqx.Module, strict=True):
    """An abstract base class for a CTF in cryo-EM."""

    @abstractmethod
    def compute_aberration_phase_shifts(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        wavelength_in_angstroms: FloatLike,
        defocus_offset: FloatLike | None = None,
        parity: Parity = "all",
    ) -> Float[Array, "y_dim x_dim"] | None:
        """Compute the frequency-dependent phase shifts due to wave aberration,
        or their even or odd part.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            The grid of frequencies in units of inverse angstroms.
        - `wavelength_in_angstroms`:
            The wavelength of the incident electrons in Angstroms.
        - `defocus_offset`:
            An optional defocus offset, in Angstroms, to apply at runtime.
        - `parity`:
            Whether to return `'all'` of the phase shifts, or their `'even'` or
            `'odd'` part in the frequency. The odd part is `None` if the CTF has no
            odd aberrations.
        """
        raise NotImplementedError

    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        wavelength_in_angstroms: FloatLike,
        amplitude_contrast_ratio: FloatLike = 0.1,
        phase_shift: FloatLike = 0.0,
        outputs_exp: bool = False,
        defocus_offset: FloatLike | None = None,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        """Compute the CTF as a JAX array.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            The grid of frequencies in units of inverse angstroms. This can
            be computed with [`cryojax.ndimage.make_frequency_grid`][]
        - `wavelength_in_angstroms`:
            The wavelength of the incident electrons in Angstroms. This
            can be retrieved from the accelerating voltage using the function
            the function [`cryojax.constants.wavelength_from_kilovolts`][].
        - `amplitude_contrast_ratio`:
            The amplitude contrast ratio. This argument is not used if `outputs_exp = True`, as
            the amplitude contrast ratio cannot simply be absorbed into a phase shift.
        - `phase_shift`:
            Additional constant phase shift applied to the frequency-dependent phase shifts.
        - `outputs_exp`:
            If `False`, return the CTF used in linear image formation theory. If `True`, return
            the CTF (or wave transfer function) as a complex exponential.

        With odd aberrations and `outputs_exp = False`, the CTF is complex.
        """  # noqa: E501
        # Get the wavelength
        wavelength_in_angstroms = jnp.asarray(wavelength_in_angstroms, dtype=float)
        # Constant phase shift, convert degrees to radians
        phase_shift = jnp.deg2rad(phase_shift)
        if not outputs_exp:
            # Compute the CTF from the even part of the phase shifts; the odd part,
            # if any, is a pure phase factor
            aberration_phase_shifts = self.compute_aberration_phase_shifts(
                frequency_grid_in_angstroms,
                wavelength_in_angstroms=wavelength_in_angstroms,
                defocus_offset=defocus_offset,
                parity="even",
            )
            odd_phase_shifts = self.compute_aberration_phase_shifts(
                frequency_grid_in_angstroms,
                wavelength_in_angstroms=wavelength_in_angstroms,
                defocus_offset=defocus_offset,
                parity="odd",
            )
            amplitude_contrast_phase_shift = compute_amplitude_contrast_phase_shift(
                jnp.asarray(amplitude_contrast_ratio, dtype=float)
            )
            ctf = jnp.sin(
                aberration_phase_shifts - (phase_shift + amplitude_contrast_phase_shift)
            )
            if odd_phase_shifts is None:
                return ctf
            return ctf * jnp.exp(-1.0j * odd_phase_shifts)
        else:
            # Compute the "complex CTF", correcting for the amplitude contrast
            # and additional phase shift in the zero mode
            aberration_phase_shifts = self.compute_aberration_phase_shifts(
                frequency_grid_in_angstroms,
                wavelength_in_angstroms=wavelength_in_angstroms,
                defocus_offset=defocus_offset,
            )
            return jnp.exp(-1.0j * (aberration_phase_shifts - phase_shift))


class AstigmaticCTF(AbstractCTF, strict=True):
    """Compute an astigmatic Contrast Transfer Function (CTF) with a
    spherical aberration correction and amplitude contrast ratio.

    !!! info
        `cryojax` uses a convention different from CTFFIND for
        astigmatism parameters. CTFFIND returns defocus major and minor
        axes, called "defocus1" and "defocus2". In order to convert
        from CTFFIND to `cryojax`,

        ```python
        defocus1, defocus2 = ... # Read from CTFFIND
        ctf = AstigmaticCTF(
            defocus_in_angstroms=(defocus1+defocus2)/2,
            astigmatism_in_angstroms=defocus1-defocus2,
            ...
        )
        ```
    """

    defocus_in_angstroms: Float[NDArrayLike, "..."]
    astigmatism_in_angstroms: Float[NDArrayLike, "..."]
    astigmatism_angle: Float[NDArrayLike, "..."]
    spherical_aberration_in_mm: Float[NDArrayLike, "..."]

    def __init__(
        self,
        defocus_in_angstroms: FloatLike = 10000.0,
        astigmatism_in_angstroms: FloatLike = 0.0,
        astigmatism_angle: FloatLike = 0.0,
        spherical_aberration_in_mm: FloatLike = 2.7,
    ):
        """**Arguments:**

        - `defocus_in_angstroms`: The mean defocus in Angstroms.
        - `astigmatism_in_angstroms`: The amount of astigmatism in Angstroms.
        - `astigmatism_angle`: The defocus angle.
        - `spherical_aberration_in_mm`: The spherical aberration coefficient in mm.
        """
        self.defocus_in_angstroms = leaf_asarray(defocus_in_angstroms, dtype=float)
        self.astigmatism_in_angstroms = leaf_asarray(
            astigmatism_in_angstroms, dtype=float
        )
        self.astigmatism_angle = leaf_asarray(astigmatism_angle, dtype=float)
        self.spherical_aberration_in_mm = leaf_asarray(
            spherical_aberration_in_mm, dtype=float
        )

    @property
    def defocus_in_um(self) -> Float[Array, "..."]:
        """The mean defocus in microns."""
        return jnp.asarray(self.defocus_in_angstroms) / 1e4

    @property
    def astigmatism_xy_in_um(self) -> Float[Array, "... 2"]:
        """The astigmatism in microns, as the vector
        `astigmatism * (cos(2 * angle), sin(2 * angle))`."""
        return (
            _polar_to_xy(self.astigmatism_in_angstroms, self.astigmatism_angle, 2) / 1e4
        )

    def compute_aberration_phase_shifts(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        wavelength_in_angstroms: FloatLike,
        defocus_offset: FloatLike | None = None,
        parity: Parity = "all",
    ) -> Float[Array, "y_dim x_dim"] | None:
        """Compute the frequency-dependent phase shifts due to wave aberration.

        This is often denoted as $\\chi(\\boldsymbol{q})$ for the in-plane
        spatial frequency $\\boldsymbol{q}$.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            The grid of frequencies in units of inverse angstroms. This can
            be computed with [`cryojax.ndimage.make_frequency_grid`][]
        - `wavelength_in_angstroms`:
            The wavelength of the incident electrons in Angstroms. This
            can be retrieved from the accelerating voltage using the function
            the function [`cryojax.constants.wavelength_from_kilovolts`][].
        - `defocus_offset`:
            An optional defocus offset to apply to the `defocus_in_angstroms` at runtime.
        - `parity`:
            The phase shifts are even, so `'odd'` returns `None`.
        """
        _check_parity(self, parity)
        if parity == "odd":
            return None
        defocus_in_angstroms = jnp.asarray(self.defocus_in_angstroms)
        # Convert spherical abberation coefficient to angstroms
        spherical_aberration_in_angstroms = (
            error_if_negative(jnp.asarray(self.spherical_aberration_in_mm)) * 1e7
        )
        # Compute phase shifts for CTF
        phase_shifts = compute_even_aberration_phase_shifts(
            frequency_grid_in_angstroms,
            jnp.asarray(wavelength_in_angstroms, dtype=float),
            (
                defocus_in_angstroms
                if defocus_offset is None
                else defocus_in_angstroms + jnp.asarray(defocus_offset, dtype=float)
            ),
            _polar_to_xy(self.astigmatism_in_angstroms, self.astigmatism_angle, 2),
            spherical_aberration_in_angstroms,
        )
        return phase_shifts


class AberratedCTF(AbstractCTF, strict=True):
    """A Contrast Transfer Function (CTF) with Cartesian astigmatism, spherical
    aberration, and optional odd aberrations (axial coma and trefoil).

    Each m-fold aberration is a vector `magnitude * (cos(m * angle), sin(m * angle))`,
    with `angle` measured as for `AstigmaticCTF.astigmatism_angle`. The phase shifts
    are then linear in every parameter:

    $$\\chi_{even} = 2 \\pi [-\\frac{1}{2} \\lambda (\\Delta f |\\boldsymbol{q}|^2
    + \\frac{1}{2} \\boldsymbol{a} \\cdot H_2(\\boldsymbol{q}))
    + \\frac{1}{4} C_s \\lambda^3 |\\boldsymbol{q}|^4],$$

    $$\\chi_{odd} = 2 \\pi \\lambda^2 |\\boldsymbol{q}|^2
    \\boldsymbol{b} \\cdot H_1(\\boldsymbol{q})
    + \\frac{2 \\pi}{3} \\lambda^2 \\boldsymbol{t} \\cdot H_3(\\boldsymbol{q}),$$

    where $H_m(\\boldsymbol{q})$ is the real and imaginary part of $(q_y + i q_x)^m$,
    for astigmatism $\\boldsymbol{a}$, coma $\\boldsymbol{b}$, and trefoil
    $\\boldsymbol{t}$.

    !!! info
        The coma from a beam tilt in mrad has magnitude, in microns,
        `spherical_aberration_in_mm` times the tilt.
    """

    defocus_in_um: Float[NDArrayLike, "..."]
    astigmatism_xy_in_um: Float[NDArrayLike, "... 2"]
    spherical_aberration_in_mm: Float[NDArrayLike, "..."]
    coma_xy_in_um: Float[NDArrayLike, "... 2"] | None
    trefoil_xy_in_um: Float[NDArrayLike, "... 2"] | None

    def __init__(
        self,
        defocus_in_um: FloatLike = 1.0,
        astigmatism_xy_in_um: Float[NDArrayLike, "... 2"] | Sequence[float] = (0.0, 0.0),
        spherical_aberration_in_mm: FloatLike = 2.7,
        coma_xy_in_um: Float[NDArrayLike, "... 2"] | Sequence[float] | None = None,
        trefoil_xy_in_um: Float[NDArrayLike, "... 2"] | Sequence[float] | None = None,
    ):
        """**Arguments:**

        - `defocus_in_um`: The mean defocus in microns.
        - `astigmatism_xy_in_um`:
            The astigmatism in microns, as a vector with fold `m = 2`.
        - `spherical_aberration_in_mm`: The spherical aberration coefficient in mm.
        - `coma_xy_in_um`:
            The axial coma in microns, as a vector with fold `m = 1`. If `None`,
            there is no coma.
        - `trefoil_xy_in_um`:
            The trefoil in microns, as a vector with fold `m = 3`. If `None`, there
            is no trefoil.
        """
        self.defocus_in_um = leaf_asarray(defocus_in_um, dtype=float)
        self.astigmatism_xy_in_um = _xy_leaf_asarray(
            astigmatism_xy_in_um, "astigmatism_xy_in_um"
        )
        self.spherical_aberration_in_mm = leaf_asarray(
            spherical_aberration_in_mm, dtype=float
        )
        self.coma_xy_in_um = (
            None
            if coma_xy_in_um is None
            else _xy_leaf_asarray(coma_xy_in_um, "coma_xy_in_um")
        )
        self.trefoil_xy_in_um = (
            None
            if trefoil_xy_in_um is None
            else _xy_leaf_asarray(trefoil_xy_in_um, "trefoil_xy_in_um")
        )

    @classmethod
    def from_polar_coordinates(
        cls,
        defocus_in_um: FloatLike = 1.0,
        astigmatism_in_um: FloatLike = 0.0,
        astigmatism_angle: FloatLike = 0.0,
        spherical_aberration_in_mm: FloatLike = 2.7,
        coma_in_um: FloatLike | None = None,
        coma_angle: FloatLike = 0.0,
        trefoil_in_um: FloatLike | None = None,
        trefoil_angle: FloatLike = 0.0,
    ) -> "AberratedCTF":
        """Construct an `AberratedCTF` from aberration magnitudes and angles, with
        angles in degrees.

        **Arguments:**

        - `defocus_in_um`: The mean defocus in microns.
        - `astigmatism_in_um`: The amount of astigmatism in microns.
        - `astigmatism_angle`: The astigmatism angle.
        - `spherical_aberration_in_mm`: The spherical aberration coefficient in mm.
        - `coma_in_um`: The amount of axial coma in microns. If `None`, there is no coma.
        - `coma_angle`: The coma angle.
        - `trefoil_in_um`:
            The amount of trefoil in microns. If `None`, there is no trefoil.
        - `trefoil_angle`: The trefoil angle.
        """
        return cls(
            defocus_in_um=defocus_in_um,
            astigmatism_xy_in_um=_polar_to_xy(astigmatism_in_um, astigmatism_angle, 2),
            spherical_aberration_in_mm=spherical_aberration_in_mm,
            coma_xy_in_um=(
                None if coma_in_um is None else _polar_to_xy(coma_in_um, coma_angle, 1)
            ),
            trefoil_xy_in_um=(
                None
                if trefoil_in_um is None
                else _polar_to_xy(trefoil_in_um, trefoil_angle, 3)
            ),
        )

    def compute_aberration_phase_shifts(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        wavelength_in_angstroms: FloatLike,
        defocus_offset: FloatLike | None = None,
        parity: Parity = "all",
    ) -> Float[Array, "y_dim x_dim"] | None:
        """Compute the frequency-dependent phase shifts due to wave aberration.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            The grid of frequencies in units of inverse angstroms. This can
            be computed with [`cryojax.ndimage.make_frequency_grid`][].
        - `wavelength_in_angstroms`:
            The wavelength of the incident electrons in Angstroms. This can be
            computed with [`cryojax.constants.wavelength_from_kilovolts`][].
        - `defocus_offset`:
            An optional defocus offset, in Angstroms, to apply at runtime.
        - `parity`:
            `'even'` for the defocus, astigmatism, and spherical aberration, `'odd'`
            for the coma and trefoil, or `'all'`. The odd part is `None` if there is
            neither coma nor trefoil.
        """
        _check_parity(self, parity)
        wavelength_in_angstroms = jnp.asarray(wavelength_in_angstroms, dtype=float)
        even_phase_shifts, odd_phase_shifts = None, None
        if parity in ("all", "even"):
            defocus_in_angstroms = jnp.asarray(self.defocus_in_um) * 1e4
            if defocus_offset is not None:
                defocus_in_angstroms += jnp.asarray(defocus_offset, dtype=float)
            spherical_aberration_in_angstroms = (
                error_if_negative(jnp.asarray(self.spherical_aberration_in_mm)) * 1e7
            )
            even_phase_shifts = compute_even_aberration_phase_shifts(
                frequency_grid_in_angstroms,
                wavelength_in_angstroms,
                defocus_in_angstroms,
                jnp.asarray(self.astigmatism_xy_in_um) * 1e4,
                spherical_aberration_in_angstroms,
            )
        if parity in ("all", "odd"):
            odd_phase_shifts = compute_odd_aberration_phase_shifts(
                frequency_grid_in_angstroms,
                wavelength_in_angstroms,
                _um_to_angstroms(self.coma_xy_in_um),
                _um_to_angstroms(self.trefoil_xy_in_um),
            )
        if parity == "even":
            return even_phase_shifts
        if parity == "odd":
            return odd_phase_shifts
        if odd_phase_shifts is None:
            return even_phase_shifts
        return even_phase_shifts + odd_phase_shifts


def _check_parity(ctf: AbstractCTF, parity: str):
    if parity not in ("all", "even", "odd"):
        raise ValueError(
            "Found invalid value for "
            f"`{ctf.__class__.__name__}.compute_aberration_phase_shifts(..., "
            "parity=...)`. Values 'all', 'even', and 'odd' are supported, but got "
            f"{parity!r}."
        )


def _polar_to_xy(magnitude, angle_in_degrees, fold: int) -> Float[Array, "... 2"]:
    """`magnitude * (cos(fold * angle), sin(fold * angle))`, `angle` in degrees."""
    magnitude = jnp.asarray(magnitude, dtype=float)
    theta = fold * jnp.deg2rad(jnp.asarray(angle_in_degrees, dtype=float))
    return jnp.stack([magnitude * jnp.cos(theta), magnitude * jnp.sin(theta)], axis=-1)


def _xy_leaf_asarray(
    value: Float[NDArrayLike, "... 2"] | Sequence[float], name: str
) -> Float[NDArrayLike, "... 2"]:
    """`leaf_asarray` for a 2-vector `AberratedCTF` argument."""
    value = leaf_asarray(value, dtype=float)
    if value.ndim == 0 or value.shape[-1] != 2:
        raise ValueError(
            f"Found that `AberratedCTF(..., {name}=...)` has shape {value.shape}, "
            "but it must be a 2-vector with shape `(..., 2)`, e.g. "
            f"`AberratedCTF(..., {name}=(0.1, -0.2))`."
        )
    return value


def _um_to_angstroms(value) -> Array | None:
    return None if value is None else jnp.asarray(value) * 1e4
