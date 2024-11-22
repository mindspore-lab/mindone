# Schedulers for Stable Diffusion Inference

## Introduction

The schedule functions take in the output of a pre-trained model, a sample which the diffusion process is iterating on, and a timestep to return a denoised sample. Schedulers define the method for iteratively adding noise to an image or for updating a sample based on model outputs (removing noise). Schedulers are often defined by a noise schedule and an update rule to solve the differential equation solution.

## Summary of Schedulers

The following table summarizes different schedulers:

| scheduler | reference |
|:------|:------|
| DDPM | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) |
| DDIM | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) |
| PLMS | [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778) |
| DPM-Solver | [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) |
| DPM-Solver++ | [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095) |
| UniPC | [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2302.04867) |


## Inference with SD2.0

Example using the default **DPM-Solver++** scheduler (Refer to [Stable Diffusion 2.0-Inference](./README.md#stable-diffusion-20)).

```shell
# Text to image generation with SD2.0
python text_to_image.py --prompt "A wolf in winter" --version 2.0
```

| DPM-Solver++ #1 | DPM-Solver++ #2 | DPM-Solver++ #3  | DPM-Solver++ #4 | DPM-Solver++ #5 |
| :----: | :----: |:--------------------------------------------------------------------------------------------------------------------------------:| :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/6a84e5ae-baab-4461-91ec-14f55b16aee2" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/62c0bb93-68a8-4ff1-8bc5-0c9305931ac0" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/92441896-8786-465b-82a2-3438aad85476" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e626ae1b-8fa6-4ad8-a083-ec39e4944277" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/d3728d81-4a31-4d53-baf0-cd79b6046ca3" width="155" height="155" /> |


 The optimal Hyperparameters can vary for different schedulers. For example, the optimal `${sampling_steps}` is 50 for PLMS, DDIM and 20 for UniPC,DPM-Solver, DPM-Solver++.

```shell
# Scheduler PLMS: set ${scheduler} to plms and ${sampling_steps} to 50
# Scheduler DDIM: set ${scheduler} to ddim and ${sampling_steps} to 50
# Scheduler DPM-Solver: set ${scheduler} to dpm_solver and ${sampling_steps} to 20
# Scheduler DPM-Solver++: set ${sampling_steps} to 20  (Note: in this case, no need to pass the ${scheduler} argument)
# Scheduler UniPC: set ${scheduler} to uni_pc and ${sampling_steps} to 20
# ${prompt} be "A Van Gogh style oil painting of sunflower" as default.
python text_to_image.py \
    --prompt ${prompt} \
    --config configs/v2-inference.yaml \
    --version 2.0 \
    --output_path ./output/ \
    --seed 42 \
    --${scheduler} \
    --n_iter 8 \
    --n_samples 1 \
    --W 512 \
    --H 512 \
    --sampling_steps ${sampling_steps}
```

#### Examples

Now, let's execute the [command above](#321-commands-of-inference-with-different-schedulers) to generate some images with different `${prompt}` and different `${scheduler}`. All these images have a default resolution of **512x512**.

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
python text_to_image.py  --prompt "A Van Gogh style oil painting of sunflower"  --config configs/v2-inference.yaml  --version 2.0  --seed 42  --${scheduler} --n_iter 8 --n_samples 1 --sampling_steps ${sampling_steps}
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/1105da61-4f12-47d3-a008-25117fddfe68" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ba5f89e8-84a6-4805-a132-34d0aff4f91a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7adf2a87-a1ed-4963-8c00-4d70e34c820c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4cfed3e7-1dff-49f1-8399-e25593d29e83" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e7d9e51f-50f8-4ed6-9685-431b813967d1" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/9f1a5530-87ac-4fa4-adc2-3b304bfc636d" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/430cc134-16cb-4327-9b88-1bc6de99f33b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2ae82e37-f27a-4805-8d05-71c8a8f8676e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b71626a5-2d39-4c70-aee7-e68cc2c10651" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b693bcaa-479c-4fdf-adce-22afd453f975" width="155" height="155" /> |

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
python text_to_image.py  --prompt "A photo of an astronaut riding a horse on mars"  --config configs/v2-inference.yaml  --version 2.0  --seed 42  --${scheduler} --n_iter 8 --n_samples 1 --sampling_steps ${sampling_steps}
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/9c80d7fe-4709-4387-b51d-fe9b86d1e92a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/cd85949c-ff35-4328-9d49-359bf30b7e30" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/96023ae4-becb-48ea-be16-9b252ae48ade" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/5c6b7578-0111-4256-a963-c99d4cc37aba" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/78006fcd-4820-41ad-9f6b-c233b2353233" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ae6af084-7930-42fd-a91d-7aaf182f5f5b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/9fbb2419-56d6-49cb-8dee-f3c85007dfc3" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/1305a1f7-dec7-4ca7-aae1-b5bb6aa1a55d" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/00cfffb6-4d59-4769-98c4-676b6ec9d7f2" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7c928546-be38-4d0a-9cb2-4646b84abbce" width="155" height="155" /> |

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
python text_to_image.py  --prompt "A high tech solarpunk utopia in the Amazon rainforest"  --config configs/v2-inference.yaml  --version 2.0  --seed 42  --${scheduler} --n_iter 8 --n_samples 1 --sampling_steps ${sampling_steps}
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e1eeef11-0aeb-43f7-8b40-8e2a0c0e9a70" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/aa88077b-513b-4502-a236-47de21452688" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/6f733664-02d3-41da-959e-ddf8433b196c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/38057a39-277a-4226-8bf9-b98cb05e064f" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/0ea319e9-442c-4b17-96fb-bce1f38d1762" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2faa35c9-c52b-4753-afdc-ea3b24afb2d2" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4c698e0f-ddfe-44eb-bb79-1e4f97b0496c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ff80bf66-4244-4624-ad1b-074aa36f62be" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ced062af-1a40-46e7-9f27-e66cd119b3e1" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/0cd3e4c1-f9ab-4636-a819-a4278c08d991" width="155" height="155" /> |

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
python text_to_image.py  --prompt "The beautiful night view of the city has various buildings, traffic flow, and lights"  --config configs/v2-inference.yaml  --version 2.0  --seed 42  --${scheduler} --n_iter 8 --n_samples 1 --sampling_steps ${sampling_steps}
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/71658f30-d89d-4e34-9195-34e14a132d3b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/f14d487b-31ff-4063-90fc-c5f8c19bbab0" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/6669df40-d960-4247-a2d9-a2a0404bef6e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/d19a79b3-58b9-453e-a6fd-bc087073aa13" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/62677a59-bf7f-47d9-8d4b-7f3c47bf4cae" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b3fdcf9b-699d-4717-a997-c7f7fac4858e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7f328896-3765-4297-b280-c33f45b442b8" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/318ff4ea-833d-4ea9-9e27-a8ea7e2c9494" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/504e8a9b-a484-4274-9819-9d214fb58b74" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/c136d39d-d893-408e-968e-3f4c465ba3da" width="155" height="155" /> |

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
python text_to_image.py  --prompt "A pikachu fine dining with a view to the Eiffel Tower"  --config configs/v2-inference.yaml  --version 2.0  --seed 42  --${scheduler} --n_iter 8 --n_samples 1 --sampling_steps ${sampling_steps}
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/8eee6ecc-7fa0-4a06-a9fe-a26a2b9b8660" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/c7a90a16-0d20-4715-ba87-dfece253f1b7" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/f743beeb-8551-43f4-8161-04fc521912e5" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/0a9b2411-a38e-4009-816c-7682ad12c78e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/db56e390-fefa-4a5a-a7c3-92d76901f9e7" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/d0430b42-db22-429b-ba42-fcb9a6cbe801" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/3b6dd274-a8d5-4b0f-86cc-568cda16d179" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/8002fa97-aa80-47ac-9f3c-23969bc38cf2" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/53e9ae3b-40ab-449b-b1a8-2254ffa267a1" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/f309319c-43bf-4db9-920f-e60720973bf1" width="155" height="155" /> |

#### Performance

experiments are tested on ascend 910 with mindspore2.1.0 graph mode
| **scheduler** | **sampling_steps** | **time(second/image)** |
|:-------------|:------------------:|:----------------------:|
|     ddim      |         50         |         16.48s         |
|     plms      |         50         |         16.77s         |
|  dmp_solver   |         20         |         12.72s         |
| dmp_solver_pp |         20         |         13.43s         |
|    uni_pc     |         20         |         14.97s         |

## Inference based on LoRA

The difference schedulers can be used for stable diffusion model + LoRA inference (see [LoRA](./lora_finetune.md#introduction) for more information).
Users can specify schedulers in the same manner as described in chapter [Inference with Different Schedulers](#32-inference-with-different-schedulers).

#### Examples

Based on the LoRA models trained on pokemon and chinese_art datasets (see [LoRA](./lora_finetune.md#get-started) for more information), we test them using different schedulers. The base model is Stable Diffusion 2.0.

- **pokemon dataset**:

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
# Set ${prompt} to "a drawing of a blue and white cat with big eyes"
bash scripts/run_test_to_image_v2_lora.sh
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/99c2463a-5e8f-458a-962e-4d28c535d878" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/df3342c8-bfe1-424f-8c76-0aad0b7363d6" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/18b9b0b0-7dc3-437a-8f87-fc86fccfd4d7" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/26cf1959-90fc-470e-8114-14d6dc1f82e5" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/32d44ebc-bc62-4336-a836-ae7cace21e86" width="155" height="155" /> |

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
# Set ${prompt} to "a cartoon of a black and white pokemon"
bash scripts/run_test_to_image_v2_lora.sh
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/d20d5e08-a0b2-49b9-8591-26e7630af5ea" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/0cbcf1e3-85db-488c-a081-1af05f2e18ce" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ea92e161-3258-4dc2-8711-dafc521cb58c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/9a2ee45a-81cf-4996-8e92-5307f8af3966" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2917429b-85c8-4305-93b8-05e199afb372" width="155" height="155" /> |

- **chinese_art dataset**:

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
# Set ${prompt} to "a painting of a group of people sitting on a hill with trees in the background and a stream of water"
bash scripts/run_test_to_image_v2_lora.sh
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/be0edf8d-62e3-4f1a-8090-433a4e7acb04" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/18c3dd5a-ff3e-4eb1-986c-1ad5e89adbdb" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/cd438ca1-f358-4a59-99e0-d728eaac0695" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b64f1a60-6887-44d9-8a39-fb1b607af2f3" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e1923489-e73f-4d34-8641-ce482cef2ce2" width="155" height="155" /> |

```bash
# Set ${scheduler} to plms, ddim, dpm_solver and uni_pc in turn (Note: also use their respective optimal ${sampling_steps})
# Note: no need to pass ${scheduler} while using dpm_solver_pp
# Set ${prompt} to "a drawing of a village with a boat and a house in the background with a red ribbon on the bottom of the picture"
bash scripts/run_test_to_image_v2_lora.sh
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/37ddf56d-7e88-4093-919d-9a79f0b59961" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/fdab0742-2809-4dbb-ac35-2e3abda5b654" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/3ca860e7-027d-4e03-b114-488e244d48a7" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/f3b5e5f5-279b-4c03-b54f-cf6e1ff10953" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/49a5c26f-03cb-4430-935f-b4e9940f725a" width="155" height="155" /> |

#### Qualitative Comparison with Diffusers

We also show some text-to-image generation samples for the LoRA models trained by **MindOne** and **Diffusers**. The base model is Stable Diffusion 2.0.

- pokemon dataset:

```bash
${prompt}="a drawing of a black and gray dragon"
```

|        | PLMS | DDIM | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| MindOne | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7fb311b7-a0f4-4aa2-b10d-8eb2deebbcdf" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/3ebd8598-468e-4853-ab87-ec5457ae2590" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/514469e1-df82-45d6-9acc-c8d1e5ff312c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/26062b45-7c36-4277-b10a-61e2ebd23af8" width="155" height="155" /> |
| Diffusers | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/0a7fa4e5-0f62-4985-8bc7-873a7a5e61a4" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/91b2fd08-d283-49e8-96a2-db15f8e93e23" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ecc3676b-ca14-480e-8719-02694a176e9d" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/37430894-9d5c-4aa1-82d3-4b84464c5bf8" width="155" height="155" /> |

```bash
${prompt}="a cartoon panda with a leaf in its mouth"
```

|        | PLMS | DDIM | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| MindOne | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/d89031eb-9ad8-4d51-9126-d6638ee7cae7" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4302c116-ba08-4d63-ae62-163da37e634c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/41487edd-ef09-4525-b09a-42e8ff79ce5b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/3408fe57-2700-495e-8677-26ddb618b2e6" width="155" height="155" /> |
| Diffusers | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/cdcf7b06-7822-4dc2-b33a-ad26d7eed705" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/08ed029d-4ccc-40b7-a877-0a37684ee753" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/5fddf4b1-07ab-4fb6-bcfe-1406720b15a4" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/6bc6986b-88af-416d-9e20-dd3ac9a9dee0" width="155" height="155" /> |

- chinese_art dataset:

```bash
${prompt}="a painting of a landscape with a mountain in the background and a river running through it with a few people on it"
```

| | PLMS | DDIM | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| MindOne | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e3f8893a-419d-4f59-9ede-63e6ff307c9b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/70a24b3e-5439-4ccd-bd63-afbd94ecfc27" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b3a4d82a-c8c7-4f85-bd80-613e7d085a6b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/f889c713-da74-40a9-a17d-bc363d77ba7e" width="155" height="155" /> |
| Diffusers | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/00a41bc8-c38d-463d-af8a-8e204e41c14a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/f95715eb-d92f-4464-ad84-f8af554c81f2" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/3adf40cc-43cd-4dcc-b05c-71541a2a8b2e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/f9c73e9b-ea3e-4232-83b8-8b640d792896" width="155" height="155" /> |
