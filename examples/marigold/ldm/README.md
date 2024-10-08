# Introduction of this lib

This lib comes from the `ldm` of [stable_diffusion_v2](https://github.com/mindspore-lab/mindone/tree/master/examples/stable_diffusion_v2) and only make a few changes.

The changes are listed below:

1. **ldm.modules.train.lr_schedule.py** and **ldm.modules.train.dynamic_lr.py** to support the IterExponential learning rate strategy of Marigold.
2. **ldm.models.diffusion.ddpm.py** to define a `MarigoldLatentDiffusion` class to support training.
3. delete **ldm.data** because unnecessary.
