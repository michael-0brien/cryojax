from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ...constants import wavelength_from_kilovolts
from ...jax_util import FloatLike, error_if_negative
from .common_functions import (
    compute_phase_shift_from_amplitude_contrast_ratio,
    compute_phase_shifts_with_spherical_aberration,
)


class AbstractCTF(eqx.Module, strict=True):
    """An abstract base class for a CTF in cryo-EM."""

    @abstractmethod
    def compute_aberration_phase_shifts(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        wavelength_in_angstroms: FloatLike,
        defocus_offset: FloatLike | None = None,
    ) -> Float[Array, "y_dim x_dim"]:
        raise NotImplementedError

    def __call__(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        voltage_in_kilovolts: FloatLike,
        amplitude_contrast_ratio: FloatLike = 0.1,
        phase_shift: FloatLike = 0.0,
        outputs_exp: bool = False,
        defocus_offset: FloatLike | None = None,
    ) -> Float[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim"]:
        """Compute the CTF as a JAX array.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            The grid of frequencies in units of inverse angstroms. This can
            be computed with [`cryojax.coordinates.make_frequency_grid`](https://mjo22.github.io/cryojax/api/coordinates/making_coordinates/#cryojax.coordinates.make_frequency_grid)
        - `voltage_in_kilovolts`:
            The accelerating voltage of the microscope in kilovolts. This
            is converted to the wavelength of incident electrons using
            the function [`cryojax.constants.convert_keV_to_angstroms`](https://mjo22.github.io/cryojax/api/constants/units/#cryojax.constants.convert_keV_to_angstroms)
        - `amplitude_contrast_ratio`:
            The amplitude contrast ratio. This argument is not used if `outputs_exp = True`, as
            the amplitude contrast ratio cannot simply be absorbed into a phase shift.
        - `phase_shift`:
            Additional constant phase shift applied to the frequency-dependent phase shifts.
        - `outputs_exp`:
            If `False`, return the CTF used in linear image formation theory. If `True`, return
            the CTF (or wave transfer function) as a complex exponential.
        """  # noqa: E501
        # Get the wavelength
        wavelength_in_angstroms = wavelength_from_kilovolts(
            jnp.asarray(voltage_in_kilovolts)
        )
        # Frequency-dependent phase shifts
        aberration_phase_shifts = self.compute_aberration_phase_shifts(
            frequency_grid_in_angstroms,
            wavelength_in_angstroms=wavelength_in_angstroms,
            defocus_offset=defocus_offset,
        )
        # Constant phase shift, convert degrees to radians
        phase_shift = jnp.deg2rad(phase_shift)
        if not outputs_exp:
            # Compute the CTF
            amplitude_contrast_phase_shift = (
                compute_phase_shift_from_amplitude_contrast_ratio(
                    jnp.asarray(amplitude_contrast_ratio, dtype=float)
                )
            )
            return jnp.sin(
                aberration_phase_shifts - (phase_shift + amplitude_contrast_phase_shift)
            )
        else:
            # Compute the "complex CTF", correcting for the amplitude contrast
            # and additional phase shift in the zero mode
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

    defocus_in_angstroms: Float[Array, ""]
    astigmatism_in_angstroms: Float[Array, ""]
    astigmatism_angle: Float[Array, ""]
    spherical_aberration_in_mm: Float[Array, ""]

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
        self.defocus_in_angstroms = jnp.asarray(defocus_in_angstroms, dtype=float)
        self.astigmatism_in_angstroms = jnp.asarray(astigmatism_in_angstroms, dtype=float)
        self.astigmatism_angle = jnp.asarray(astigmatism_angle, dtype=float)
        self.spherical_aberration_in_mm = error_if_negative(
            jnp.asarray(spherical_aberration_in_mm, dtype=float)
        )

    def compute_aberration_phase_shifts(
        self,
        frequency_grid_in_angstroms: Float[Array, "y_dim x_dim 2"],
        wavelength_in_angstroms: FloatLike,
        defocus_offset: FloatLike | None = None,
    ) -> Float[Array, "y_dim x_dim"]:
        """Compute the frequency-dependent phase shifts due to wave aberration.

        This is often denoted as $\\chi(\\boldsymbol{q})$ for the in-plane
        spatial frequency $\\boldsymbol{q}$.

        **Arguments:**

        - `frequency_grid_in_angstroms`:
            The grid of frequencies in units of inverse angstroms. This can
            be computed with [`cryojax.coordinates.make_frequency_grid`](https://mjo22.github.io/cryojax/api/coordinates/making_coordinates/#cryojax.coordinates.make_frequency_grid)
        - `voltage_in_kilovolts`:
            The accelerating voltage of the microscope in kilovolts. This
            is converted to the wavelength of incident electrons using
            the function [`cryojax.constants.convert_keV_to_angstroms`](https://mjo22.github.io/cryojax/api/constants/units/#cryojax.constants.convert_keV_to_angstroms)
        - `defocus_offset`:
            An optional defocus offset to apply to the `defocus_in_angstroms` at runtime.
        """
        astigmatism_angle = jnp.deg2rad(self.astigmatism_angle)
        # Convert spherical abberation coefficient to angstroms
        spherical_aberration_in_angstroms = self.spherical_aberration_in_mm * 1e7
        # Compute phase shifts for CTF
        phase_shifts = compute_phase_shifts_with_spherical_aberration(
            frequency_grid_in_angstroms,
            (
                self.defocus_in_angstroms
                if defocus_offset is None
                else self.defocus_in_angstroms + jnp.asarray(defocus_offset, dtype=float)
            ),
            self.astigmatism_in_angstroms,
            astigmatism_angle,
            jnp.asarray(wavelength_in_angstroms, dtype=float),
            spherical_aberration_in_angstroms,
        )
        return phase_shifts
