# Schedulers for Stable Diffusion Inference

## Introduction
Mindone contains multiple schedule functions for the diffusion process.

The schedule functions take in the output of a pre-trained model, a sample which the diffusion process is iterating on, and a timestep to return a denoised sample. Schedulers define the method for iteratively adding noise to an image or for updating a sample based on model outputs (removing noise). Schedulers are often defined by a noise schedule and an update rule to solve the differential equation solution.

## Summary of Schedulers

Mindone implements 5 different schedulers in addition to the DDPM scheduler. The following table summarizes these schedulers:

| Scheduler | Reference |                                                                 
|:------:|:------:|
| DDPM | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) |
| DDIM | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) |
| PLMS | [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778) |
| DPM-Solver | [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) |
| DPM-Solver++ | [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095) |
| UniPC | [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2302.04867) |

## Preparation

### Fast Inference

Normally, you can test the stable diffusion model using the following command using the default PLMS scheduler:
```shell
# Text to image generation with SD2.0 
python text_to_image.py --prompt "A wolf in winter"
```

For fast inference, we provide several scripts using different schedulers:

```bash
# Text to image generation with SD2.0, using PLMS scheduler
bash scripts/tests/test_plms_sampler.sh
# Text to image generation with SD2.0, using DDIM scheduler
bash scripts/tests/test_ddim_sampler.sh
# Text to image generation with SD2.0, using DPM-Solver scheduler
bash scripts/tests/test_dpmsolver_sampler.sh
# Text to image generation with SD2.0, using DPM-Solver++ scheduler
bash scripts/tests/test_dpmsolverpp_sampler.sh
# Text to image generation with SD2.0, using UniPC scheduler
bash scripts/tests/test_unipc_sampler.sh
```

Some hyper-parameters are not shared for different schedulers, e.g., sampling steps. For instance, DPM-Solver, DPM-Solver++, and UniPC need smaller sampling steps (e.g., 20) than DDIM and PLMS samplers (e.g., 50).

Note that, if you set big sampling steps for DPM-Solver, DPM-Solver++, and UniPC schedulers, the program will report a warning such as **The selected sampling timesteps are not appropriate for UniPC sampler**.

### Visual Comparison

Some text-to-image generation examples are shown here:

```bash
A Van Gogh style oil painting of sunflower
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/1105da61-4f12-47d3-a008-25117fddfe68" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ba5f89e8-84a6-4805-a132-34d0aff4f91a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7adf2a87-a1ed-4963-8c00-4d70e34c820c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4cfed3e7-1dff-49f1-8399-e25593d29e83" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e7d9e51f-50f8-4ed6-9685-431b813967d1" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/9f1a5530-87ac-4fa4-adc2-3b304bfc636d" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/430cc134-16cb-4327-9b88-1bc6de99f33b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2ae82e37-f27a-4805-8d05-71c8a8f8676e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b71626a5-2d39-4c70-aee7-e68cc2c10651" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b693bcaa-479c-4fdf-adce-22afd453f975" width="155" height="155" /> |

```bash
A photo of an astronaut riding a horse on mars
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/9c80d7fe-4709-4387-b51d-fe9b86d1e92a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ba5f89e8-84a6-4805-a132-34d0aff4f91a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7adf2a87-a1ed-4963-8c00-4d70e34c820c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4cfed3e7-1dff-49f1-8399-e25593d29e83" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e7d9e51f-50f8-4ed6-9685-431b813967d1" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ae6af084-7930-42fd-a91d-7aaf182f5f5b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/430cc134-16cb-4327-9b88-1bc6de99f33b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2ae82e37-f27a-4805-8d05-71c8a8f8676e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b71626a5-2d39-4c70-aee7-e68cc2c10651" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b693bcaa-479c-4fdf-adce-22afd453f975" width="155" height="155" /> |

```bash
A high tech solarpunk utopia in the Amazon rainforest
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e1eeef11-0aeb-43f7-8b40-8e2a0c0e9a70" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ba5f89e8-84a6-4805-a132-34d0aff4f91a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7adf2a87-a1ed-4963-8c00-4d70e34c820c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4cfed3e7-1dff-49f1-8399-e25593d29e83" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e7d9e51f-50f8-4ed6-9685-431b813967d1" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2faa35c9-c52b-4753-afdc-ea3b24afb2d2" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/430cc134-16cb-4327-9b88-1bc6de99f33b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2ae82e37-f27a-4805-8d05-71c8a8f8676e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b71626a5-2d39-4c70-aee7-e68cc2c10651" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b693bcaa-479c-4fdf-adce-22afd453f975" width="155" height="155" /> |

```bash
The beautiful night view of the city has various buildings, traffic flow, and lights
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/71658f30-d89d-4e34-9195-34e14a132d3b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ba5f89e8-84a6-4805-a132-34d0aff4f91a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7adf2a87-a1ed-4963-8c00-4d70e34c820c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4cfed3e7-1dff-49f1-8399-e25593d29e83" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e7d9e51f-50f8-4ed6-9685-431b813967d1" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b3fdcf9b-699d-4717-a997-c7f7fac4858e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/430cc134-16cb-4327-9b88-1bc6de99f33b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2ae82e37-f27a-4805-8d05-71c8a8f8676e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b71626a5-2d39-4c70-aee7-e68cc2c10651" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b693bcaa-479c-4fdf-adce-22afd453f975" width="155" height="155" /> |

```bash
A pikachu fine dining with a view to the Eiffel Tower
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/8eee6ecc-7fa0-4a06-a9fe-a26a2b9b8660" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ba5f89e8-84a6-4805-a132-34d0aff4f91a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7adf2a87-a1ed-4963-8c00-4d70e34c820c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4cfed3e7-1dff-49f1-8399-e25593d29e83" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e7d9e51f-50f8-4ed6-9685-431b813967d1" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/d0430b42-db22-429b-ba42-fcb9a6cbe801" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/430cc134-16cb-4327-9b88-1bc6de99f33b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2ae82e37-f27a-4805-8d05-71c8a8f8676e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b71626a5-2d39-4c70-aee7-e68cc2c10651" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b693bcaa-479c-4fdf-adce-22afd453f975" width="155" height="155" /> |
