# Apply LoRA Checkpoints trained with MindSpore to Torch Inference

## Convert MS checkpoint to PT

```
python convert_lora_ms2pt.py {path to ms_ckpt}
```

The converted checkpoint will be saved in the same folder of {path to ms_ckpt}.

## Run inference in diffusers

Please specify the file path of the converted checkpoint in torch sdxl inference script. 

For demonstration, we provide an example sdxl lora inference script `diffusers_scripts/infer_lora.py` based on diffusers.

```
python diffusers_scripts/infer_lora.py --model_path {path to converted ckpt}
```

Images will be generated in the folder of "{model_path)-gen-images".

## Check consistency between PT and MS inference results (optional)
 
To check inference consistency quantitatively, you should make sure MS ant PT use the same initial latent noise and text prompt for diffusion sampling. Here are reference instructions to achieve it. 

1. Save the initial latent noise used in diffusers
    In `path/to/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`,  modify the `prepare_latents` function to save the init noise as numpy as follows. 

    ```python
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, save_noise=False):
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

2. Use the same latent noise in MS inference

    Please set `init_latent_path` and `prompt` in MS inference script referring the following script. 
    
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
