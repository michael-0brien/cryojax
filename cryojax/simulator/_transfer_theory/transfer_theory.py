import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from ..._internal import error_if_not_fractional, leaf_asarray
from ...jax_util import FloatLike, NDArrayLike
from ...ndimage import AbstractFourierOperator
from .._image_config import AbstractImageConfig
from .transfer_function import AbstractCTF


class AbstractTransferTheory(eqx.Module, strict=True):
    """A transfer theory for the weak-phase approximation. This class
    propagates the fourier spectrum of the object from a plane directly below it to
    the plane of the detector. In other terms, it computes a noiseless cryo-EM
    image from a 2D projection.
    """

    ctf: eqx.AbstractVar[AbstractCTF]


class ContrastTransferTheory(AbstractTransferTheory, strict=True):
    """A transfer theory for the weak-phase approximation. This class
    propagates the fourier spectrum of the object from a plane directly below it to
    the plane of the detector. In other terms, it computes a noiseless cryo-EM
    image from a 2D projection.
    """

    ctf: AbstractCTF
    envelope: AbstractFourierOperator | None
    amplitude_contrast_ratio: Float[NDArrayLike, "..."]
    phase_shift: Float[NDArrayLike, "..."]

    def __init__(
        self,
        ctf: AbstractCTF,
        envelope: AbstractFourierOperator | None = None,
        amplitude_contrast_ratio: FloatLike = 0.1,
        phase_shift: FloatLike = 0.0,
    ):
        """**Arguments:**

        - `ctf`: The contrast transfer function model.
        - `envelope`: The envelope function of the optics model.
        - `amplitude_contrast_ratio`: The amplitude contrast ratio.
        - `phase_shift`: The additional phase shift.
        """

        self.ctf = ctf
        self.envelope = envelope
        self.amplitude_contrast_ratio = leaf_asarray(
            amplitude_contrast_ratio, dtype=float
        )
        self.phase_shift = leaf_asarray(phase_shift, dtype=float)

    def __check_init__(self):
        if self.envelope is not None:
            if 2 not in self.envelope.spatial_dims:
                raise ValueError(
                    "`ContrastTransferTheory.envelope` of type "
                    f"`{self.envelope.__class__.__name__}` must support "
                    "`spatial_dim = 2`, but found that "
                    f"`{self.envelope.__class__.__name__}.spatial_dims = "
                    f"{self.envelope.spatial_dims}`."
                )

    def propagate_object(
        self,
        object_spectrum: (
            Complex[
                Array,
                "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}",
            ]
            | Complex[
                Array,
                "{image_config.padded_y_dim} {image_config.padded_x_dim}",
            ]
        ),
        image_config: AbstractImageConfig,
        *,
        offset_z_in_angstroms: FloatLike | None = None,
        is_ewald_sphere: bool = False,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim//2+1}"]:
        """Apply the CTF directly to the phase shifts in the exit plane.

        **Arguments:**

        - `object_spectrum`:
            The fourier spectrum of the scatterer phase shifts in a plane directly
            below it.
        - `image_config`:
            The configuration of the resulting image.
        - `is_ewald_sphere`:
            If `False`, the `object_spectrum` is a projection
            approximation and is therefore the fourier transform of a real-valued
            array. If `True`, `object_spectrum` is extracted from
            the ewald sphere and is therefore the fourier transform of a complex-valued
            array. `object_spectrum` is used as given (not conjugated).
        - `offset_z_in_angstroms`:
            An optional offset in Angstroms of the specimen along z, added to the
            defocus at runtime, e.g. the `offset_z_in_angstroms` of an `AbstractPose`.
        """
        amplitude_contrast_ratio = error_if_not_fractional(
            jnp.asarray(self.amplitude_contrast_ratio)
        )
        phase_shift = jnp.asarray(self.phase_shift)
        frequency_grid = image_config.get_frequency_grid(padding=True, physical=True)
        # Compute the CTF, including additional phase shifts
        ctf_array = self.ctf(
            frequency_grid,
            wavelength_in_angstroms=image_config.wavelength_in_angstroms,
            phase_shift=phase_shift,
            amplitude_contrast_ratio=amplitude_contrast_ratio,
            outputs_exp=False,
            offset_z_in_angstroms=offset_z_in_angstroms,
        )
        if not is_ewald_sphere:
            # ... compute the contrast as the CTF multiplied by the exit plane
            # phase shifts
            contrast_spectrum = ctf_array * object_spectrum
        else:
            # ... split the spectrum into those of the real and imaginary parts of
            # the exit plane phase shifts, on the half plane
            x_dim = image_config.padded_x_dim
            spectrum_at_q = object_spectrum[:, : x_dim // 2 + 1]
            spectrum_at_minus_q = jnp.roll(
                jnp.flip(object_spectrum, axis=(0, 1)), 1, axis=(0, 1)
            )[:, : x_dim // 2 + 1]
            real_spectrum = (spectrum_at_q + jnp.conj(spectrum_at_minus_q)) / 2
            imaginary_spectrum = (spectrum_at_q - jnp.conj(spectrum_at_minus_q)) / 2j
            # ... and transfer the imaginary part with the CTF at a phase shift
            # lowered by 90 degrees
            imaginary_ctf_array = self.ctf(
                frequency_grid,
                wavelength_in_angstroms=image_config.wavelength_in_angstroms,
                phase_shift=phase_shift - 90.0,
                amplitude_contrast_ratio=amplitude_contrast_ratio,
                outputs_exp=False,
                offset_z_in_angstroms=offset_z_in_angstroms,
            )
            contrast_spectrum = (
                ctf_array * real_spectrum - imaginary_ctf_array * imaginary_spectrum
            )
        if self.envelope is not None:
            contrast_spectrum *= self.envelope(frequency_grid)

        return contrast_spectrum


class WaveTransferTheory(AbstractTransferTheory, strict=True):
    """An optics model that propagates the exit wave to the detector plane."""

    ctf: AbstractCTF

    def __init__(
        self,
        ctf: AbstractCTF,
    ):
        """**Arguments:**

        - `ctf`: The contrast transfer function model.
        """

        self.ctf = ctf

    def propagate_exit_wave(
        self,
        wavefunction_spectrum: Complex[
            Array,
            "{image_config.padded_y_dim} {image_config.padded_x_dim}",
        ],
        image_config: AbstractImageConfig,
        *,
        offset_z_in_angstroms: FloatLike | None = None,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        """Apply the wave transfer function to the wavefunction in the exit plane.

        **Arguments:**

        - `wavefunction_spectrum`:
            The fourier spectrum of the wavefunction in the exit plane.
        - `image_config`:
            The configuration of the resulting image.
        - `offset_z_in_angstroms`:
            An optional offset in Angstroms of the specimen along z, added to the
            defocus at runtime, e.g. the `offset_z_in_angstroms` of an `AbstractPose`.
        """
        frequency_grid = image_config.get_frequency_grid(padding=True, full=True)
        # Compute the wave transfer function
        ctf_array = self.ctf(
            frequency_grid,
            wavelength_in_angstroms=image_config.wavelength_in_angstroms,
            outputs_exp=True,
            offset_z_in_angstroms=offset_z_in_angstroms,
        )
        # ... compute the contrast as the CTF multiplied by the exit plane
        # phase shifts
        wavefunction_spectrum = ctf_array * wavefunction_spectrum

        return wavefunction_spectrum
