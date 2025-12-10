# Image formation models

???+ abstract "`cryojax.simulator.AbstractImageModel`"
    ::: cryojax.simulator.AbstractImageModel
        options:
            members:
                - raw_simulate
                - image_config
                - pose
                - signal_region

::: cryojax.simulator.LinearImageModel
        options:
            members:
                - __init__
                - simulate
                - raw_simulate
                - postprocess

---

::: cryojax.simulator.ProjectionImageModel
        options:
            members:
                - __init__
                - simulate
                - raw_simulate
                - postprocess


---

::: cryojax.simulator.ContrastImageModel
        options:
            members:
                - __init__
                - simulate
                - raw_simulate
                - postprocess

---

::: cryojax.simulator.IntensityImageModel
        options:
            members:
                - __init__
                - simulate
                - raw_simulate
                - postprocess

---

::: cryojax.simulator.ElectronCountsImageModel
        options:
            members:
                - __init__
                - simulate
                - raw_simulate
                - postprocess
