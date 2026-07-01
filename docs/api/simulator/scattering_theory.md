# Scattering approximations

??? abstract "`cryojax.simulator.AbstractScatteringTheory`"
    ::: cryojax.simulator.AbstractScatteringTheory
        options:
            members:
                - compute_contrast_spectrum
                - compute_intensity_spectrum

# Weak-phase approximation

::: cryojax.simulator.WeakPhaseScatteringTheory
        options:
            members:
                - __init__
                - compute_object_spectrum
                - compute_contrast_spectrum
                - compute_intensity_spectrum

# Explicit wavefunction methods

??? abstract "`cryojax.simulator.AbstractWaveScatteringTheory`"
    ::: cryojax.simulator.AbstractScatteringTheory
        options:
            members:
                - transfer_theory
                - compute_exit_wave

::: cryojax.simulator.RytovScatteringTheory
        options:
            members:
                - __init__
                - compute_exit_wave
                - compute_contrast_spectrum
                - compute_intensity_spectrum
