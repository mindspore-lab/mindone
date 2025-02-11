# Use MindSpore LoRA Checkpoints for Torch Inference

## 1. Convert MS checkpoint to PT

To convert the fine-tuned LoRA checkpoint, which is ~24MB for rank=4, please run as follows.

```
python convert_lora_ms2pt.py --ms_ckpt {path to mindspore lora ckpt}
```

The converted checkpoint will be saved in "{ms_ckpt}_pt.ckpt".

## 2. Run inference in diffusers

For Torch inference, you can load the converted LoRA checkpoint in your inference code without further adjustment.

For demonstration, we provide an example sdxl lora inference script `diffusers_scripts/infer_lora.py` based on diffusers.

```
python diffusers_scripts/infer_lora.py --model_path {path to converted lora ckpt} --prompt {prompt}
```

Images will be generated in the folder of "{model_path)-gen-images".

## 3. Check consistency between PT and MS inference results (optional)

To check inference consistency quantitatively, you should make sure MS ant PT use the same initial latent noise and text prompt for diffusion sampling. Here are reference instructions to achieve it.

- Save the initial latent noise used in diffusers
    In `path/to/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`,  modify the `prepare_latents` function to save the init noise as numpy as follows.

    ```python
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, save_noise=True):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if save_noise:
            import numpy as np
            save_fp = '/tmp/sdxl_init_latents.npy'
            np.save(save_fp, latents.cpu().numpy())

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    ```

	The initial latent noise will be saved in /tmp/sdxl_init_latents.npy

- Use the same latent noise in MS inference

    Please set `init_latent_path` and `prompt` in MS inference script referring to the following script.

    ```shell
    base_ckpt_path='models/sd_xl_base_1.0_ms.ckpt'
    ckpt_path="lora_ft_pokemon/SDXL-base-1.0_40000_lora.ckpt"
    init_latent_path='/home/hyx/diffusers_sdxl_noise.npy'

    python demo/sampling_without_streamlit.py \
      --task txt2img \
      --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
      --save_path outputs/lora_pokemon_fix_noise \
      --weight $base_ckpt_path,$ckpt_path \
      --prompt "a black and white photo of a butterfly" \
      --init_latent_path $init_latent_path \
      --device_target Ascend \
      --discretization "DiffusersDDPMDiscretization" \
      --precision_keep_origin_dtype True \
    ```

    Note that if you use diffusers for comparison, you should set discretization and precision mode as above to minimize the computational difference.

## Results

Here are some generation results for comparison between MS and PT LoRA inference, where the LoRA checkpoint is derived by fine-tuning on the Pokemon dataset using MindONE.

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/3b664498-f82d-49a9-ad06-876647579d15" width="30%" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/e761ba93-bf97-4bc3-a6d1-4caccdd1614d" width="30%" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/0ef7b3e2-0582-4856-bff5-51b95b9503ee" width="30%" />
</div>
<p align="center">
  <em> MindSpore generation results using the LoRA checkpoint fine-tuned on Pokemon dataset </em>
</p>

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/040c455b-21bd-4bf0-8818-d7378e55d67c" width="30%" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/c72272e7-9757-4667-ae9d-7e1115ddc56d" width="30%" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/02654e25-ed6b-41dd-830a-bd2b63d04d84" width="30%" />
</div>
<p align="center">
  <em> Torch(diffusers) generation results using the same LoRA checkpoint </em>
</p>

The generated images for MS and PT are highly consistent as we can see. Quantitatively, the average absolute pixel error between MS and PT-generated images is below 5.
