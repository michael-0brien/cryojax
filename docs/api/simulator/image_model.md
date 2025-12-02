# Image formation models

???+ abstract "`cryojax.simulator.AbstractImageModel`"
    ::: cryojax.simulator.AbstractImageModel
        options:
            members:
                - compute_fourier_image
                - get_pose
                - get_signal_region
                - get_image_config

::: cryojax.simulator.LinearImageModel
        options:
            members:
                - __init__
                - simulate
                - postprocess

---

::: cryojax.simulator.ProjectionImageModel
        options:
            members:
                - __init__
                - simulate
                - postprocess


---

::: cryojax.simulator.ContrastImageModel
        options:
            members:
                - __init__
                - simulate
                - postprocess

---

::: cryojax.simulator.IntensityImageModel
        options:
            members:
                - __init__
                - simulate
                - postprocess

---

::: cryojax.simulator.ElectronCountsImageModel
        options:
            members:
                - __init__
                - simulate
                - postprocess
