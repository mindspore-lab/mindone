
# Usage

## Convert MS checkpoint to PT

```
python convert_lora_ms2pt.py {path to ms_ckpt}
```

The converted checkpoint will be saved in the same folder of {path to ms_ckpt}.

## Run inference in diffusers

Specify the model path with the converted checkpoint in diffusers

```
python infer_lora.py --model_path {path to converted ckpt}
```

### Test cosistency between MS and PT inference with LoRA

1. Get fixed noise from diffusers

    Hack `/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py` (in the prepare_latents function) to save the init noise as numpy. 

2. Set fixed noise in MS 

    Hack `gm/models/diffusion.py`, modify do_sample function, set 
    ```
    load_noise=True,
    noise_fp='path to the fixed noise'
    ```   

3. Inference


```shell
export MS_PYNATIVE_GE=1
export PYTHONPATH=$(pwd):$PYTHONPATH

base_ckpt_path='models/sd_xl_base_1.0_ms.ckpt'
ckpt_path="lora_ft_pokemon/SDXL-base-1.0_40000_lora.ckpt"

python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --save_path outputs/lora_pokemon_fix_noise \
  --weight $base_ckpt_path,$ckpt_path \
  --prompt "a black and white photo of a butterfly" \
  --device_target Ascend \
  --discretization "DiffusersDDPMDiscretization" \
  --seed 43 \
 
```

Make sure to set  precision_mode="must_keep_origin_dtype" for precision alignment.
```
ms.context.set_context(mode=args.ms_mode, device_target=args.device_target, ascend_config=dict(precision_mode="must_keep_origin_dtype")) # NOTE: Needed for aligning with diffusers
```

