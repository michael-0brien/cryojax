import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Inexact

from ..._internal import error_if_not_fractional, leaf_asarray
from ...jax_util import FloatLike, NDArrayLike
from ...ndimage import AbstractFourierOperator, nufft_resample
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
            | Inexact[
                Array,
                "{image_config.padded_y_dim} {image_config.padded_x_dim}",
            ]
        ),
        image_config: AbstractImageConfig,
        *,
        defocus_offset: FloatLike | None = None,
        is_ewald_sphere: bool = False,
        is_real_space: bool = False,
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
            array.
        - `defocus_offset`:
            An optional defocus offset to apply to the CTF defocus at
            runtime.
        - `is_real_space`:
            If `True`, `object_spectrum` is the object in real space.
        """
        object_spectrum = _magnify_object(
            object_spectrum,
            image_config,
            is_real_space=is_real_space,
            is_complex_object=is_ewald_sphere,
        )
        amplitude_contrast_ratio = error_if_not_fractional(
            jnp.asarray(self.amplitude_contrast_ratio)
        )
        phase_shift = jnp.asarray(self.phase_shift)
        frequency_grid = image_config.get_frequency_grid(padding=True, physical=True)
        if not is_ewald_sphere:
            # Compute the CTF, including additional phase shifts
            ctf_array = self.ctf(
                frequency_grid,
                wavelength_in_angstroms=image_config.wavelength_in_angstroms,
                phase_shift=phase_shift,
                amplitude_contrast_ratio=amplitude_contrast_ratio,
                outputs_exp=False,
                defocus_offset=defocus_offset,
            )
            # ... compute the contrast as the CTF multiplied by the exit plane
            # phase shifts
            contrast_spectrum = ctf_array * object_spectrum
        else:
            # The kernel below evaluates the phase shifts at `q` for both the `q` and
            # `-q` terms, which is only correct for even aberrations.
            if (
                self.ctf.compute_aberration_phase_shifts(
                    frequency_grid,
                    wavelength_in_angstroms=image_config.wavelength_in_angstroms,
                    parity="odd",
                )
                is not None
            ):
                ctf_name = self.ctf.__class__.__name__
                raise NotImplementedError(
                    f"Found that `ContrastTransferTheory(ctf={ctf_name}(...))` has odd "
                    "aberrations (e.g. `AberratedCTF(coma_xy_in_um=...)` or "
                    "`AberratedCTF(trefoil_xy_in_um=...)`), but it was used with a "
                    "volume integrator that outputs the Ewald sphere (e.g. "
                    "`EwaldSphereExtraction`). Odd aberrations are not yet supported "
                    "with the Ewald sphere. Either instantiate the CTF without odd "
                    "aberrations, or use a volume integrator that outputs a "
                    "projection (e.g. `FourierSliceExtraction`)."
                )
            # Propagate to the exit plane when the phase spectrum is
            # the surface of the ewald sphere
            aberration_phase_shifts = self.ctf.compute_aberration_phase_shifts(
                frequency_grid,
                wavelength_in_angstroms=image_config.wavelength_in_angstroms,
                defocus_offset=defocus_offset,
            ) - jnp.deg2rad(phase_shift)
            contrast_spectrum = _compute_contrast_from_ewald_sphere(
                object_spectrum,
                aberration_phase_shifts,
                amplitude_contrast_ratio,
                image_config,
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
        defocus_offset: FloatLike | None = None,
        is_real_space: bool = False,
    ) -> Complex[Array, "{image_config.padded_y_dim} {image_config.padded_x_dim}"]:
        """Apply the wave transfer function to the wavefunction in the exit plane.

        **Arguments:**

        - `wavefunction_spectrum`:
            The fourier spectrum of the wavefunction in the exit plane.
        - `image_config`:
            The configuration of the resulting image.
        - `defocus_offset`:
            An optional defocus offset to apply to the CTF defocus at
            runtime.
        - `is_real_space`:
            If `True`, `wavefunction_spectrum` is the wavefunction in real space.
        """
        if image_config.is_anisotropic:
            # Resample the scattered wave, which is compact, rather than the incident
            # plane wave, which is not
            wavefunction = (
                wavefunction_spectrum
                if is_real_space
                else jnp.fft.ifftn(wavefunction_spectrum)
            )
            wavefunction_spectrum = nufft_resample(
                wavefunction - 1.0,
                image_config.magnification_matrix,
                outputs_rfft=False,
                **image_config.options.get("nufft", {}),
            )
            y_dim, x_dim = image_config.padded_shape
            is_origin = (jnp.arange(y_dim) == 0)[:, None] & (jnp.arange(x_dim) == 0)[
                None, :
            ]
            wavefunction_spectrum += image_config.padded_n_pixels * is_origin
        elif is_real_space:
            wavefunction_spectrum = jnp.fft.fftn(wavefunction_spectrum)
        frequency_grid = image_config.get_frequency_grid(padding=True, full=True)
        # Compute the wave transfer function
        ctf_array = self.ctf(
            frequency_grid,
            wavelength_in_angstroms=image_config.wavelength_in_angstroms,
            outputs_exp=True,
            defocus_offset=defocus_offset,
        )
        # ... compute the contrast as the CTF multiplied by the exit plane
        # phase shifts
        wavefunction_spectrum = ctf_array * wavefunction_spectrum

        return wavefunction_spectrum


def _magnify_object(
    object: Inexact[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim//2+1"],
    image_config: AbstractImageConfig,
    *,
    is_real_space: bool,
    is_complex_object: bool,
) -> Complex[Array, "y_dim x_dim"] | Complex[Array, "y_dim x_dim//2+1"]:
    """Magnify an object onto the detector, returning its spectrum: resampled through
    the anisotropic magnification, if any, as the full plane for a complex object and
    otherwise the half plane."""
    shape = image_config.padded_shape
    if image_config.is_anisotropic:
        if not is_real_space:
            object = (
                jnp.fft.ifftn(object, s=shape)
                if is_complex_object
                else jnp.fft.irfftn(object, s=shape)
            )
        return nufft_resample(
            object,
            image_config.magnification_matrix,
            outputs_rfft=not is_complex_object,
            **image_config.options.get("nufft", {}),
        )
    if is_real_space:
        return jnp.fft.fftn(object) if is_complex_object else jnp.fft.rfftn(object)
    return object


def _compute_contrast_from_ewald_sphere(
    object_spectrum,
    aberration_phase_shifts,
    amplitude_contrast_ratio,
    image_config,
):
    cos, sin = jnp.cos(aberration_phase_shifts), jnp.sin(aberration_phase_shifts)
    ac = amplitude_contrast_ratio
    # Compute the contrast, breaking the computation into positive and
    # negative frequencies
    y_dim, x_dim = image_config.padded_y_dim, image_config.padded_x_dim
    # ... first handle the grid of frequencies
    pos_object_yx = object_spectrum[1:, 1 : x_dim // 2 + x_dim % 2]
    neg_object_yx = jnp.flip(
        jnp.flip(object_spectrum[1:, x_dim // 2 + x_dim % 2 :], axis=-1),
        axis=0,
    )
    if x_dim % 2 == 0:
        pos_object_yx = jnp.concatenate(
            (pos_object_yx, neg_object_yx[:, -1, None].conj()), axis=-1
        )
    contrast_yx = _ewald_propagate_kernel(
        pos_object_yx,
        neg_object_yx,
        ac,
        sin[1:, 1:],
        cos[1:, 1:],
    )
    # ... next handle the line of frequencies at y = 0
    pos_object_0x = object_spectrum[0, 1 : x_dim // 2 + x_dim % 2]
    neg_object_0x = jnp.flip(object_spectrum[0, x_dim // 2 + x_dim % 2 :], axis=-1)
    if x_dim % 2 == 0:
        pos_object_0x = jnp.concatenate(
            (pos_object_0x, neg_object_0x[-1, None].conj()), axis=0
        )
    contrast_0x = _ewald_propagate_kernel(
        pos_object_0x,
        neg_object_0x,
        ac,
        sin[0, 1 : x_dim // 2 + 1],
        cos[0, 1 : x_dim // 2 + 1],
    )
    # ... then handle the line of frequencies at x = 0
    pos_object_y0 = object_spectrum[1 : y_dim // 2 + y_dim % 2, 0]
    neg_object_y0 = jnp.flip(object_spectrum[y_dim // 2 + y_dim % 2 :, 0], axis=-1)
    if y_dim % 2 == 0:
        pos_object_y0 = jnp.concatenate(
            (pos_object_y0, neg_object_y0[-1, None].conj()), axis=0
        )
    contrast_y0 = _ewald_propagate_kernel(
        pos_object_y0,
        neg_object_y0,
        ac,
        sin[1 : y_dim // 2 + 1, 0],
        cos[1 : y_dim // 2 + 1, 0],
    )
    # ... concatenate the zero mode to the line of frequencies at x = 0
    object_00 = object_spectrum[0, 0]
    contrast_00 = _ewald_propagate_kernel(
        object_00,
        object_00,
        ac,
        sin[0, 0],
        cos[0, 0],
    )
    contrast_y0 = jnp.concatenate(
        (
            contrast_00[None],
            (contrast_y0 if y_dim % 2 == 1 else contrast_y0[:-1]),
            jnp.flip(contrast_y0.conjugate()),
        ),
        axis=0,
    )
    # ... concatenate the results
    contrast_yx = jnp.concatenate((contrast_0x[None, :], contrast_yx), axis=0)
    contrast_spectrum = jnp.concatenate((contrast_y0[:, None], contrast_yx), axis=1)

    return contrast_spectrum


def _ewald_propagate_kernel(pos, neg, ac, sin, cos):
    w1, w2 = ac, jnp.sqrt(1 - ac**2)
    return 0.5 * (
        (w2 * (pos.real + neg.real) + w1 * (pos.imag + neg.imag)) * sin
        + (w2 * (pos.imag + neg.imag) - w1 * (pos.real + neg.real)) * cos
        + 1.0j
        * (
            (w2 * (pos.imag - neg.imag) + w1 * (neg.real - pos.real)) * sin
            + (w2 * (neg.real - pos.real) + w1 * (neg.imag - pos.imag)) * cos
        )
    )
