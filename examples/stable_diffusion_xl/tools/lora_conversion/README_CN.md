# 在Torch中使用MindSpore LoRA训练权重进行推理

## 1. 模型权重转换

为了将MindSpore LoRA微调的权重转换成Torch推理可用的权重，可运行如下脚本。

```
python convert_lora_ms2pt.py --ms_ckpt {path to mindspore lora ckpt}
```

转换后的 checkpoint 默认保存路径为"{ms_ckpt}_pt.ckpt"。转换后的权重只有LoRA的参数，因此很小，在rank=4的情况下，约占24MB。

## 2. 加载权重推理

在Torch上推理时，只需要指定转换后的LoRA权重路径进行加载（SDXL的网络权重不变）。
作为示例，我们提供了一个基于 diffusers实现的sdxl lora推理脚本`diffusers_scripts/infer_lora.py`，在该脚本中对加载MS转Torch后的LoRA权重，运行命令如下。

```
python diffusers_scripts/infer_lora.py --model_path {path to converted lora ckpt} --prompt {prompt}
```

生成的图像将保存在文件夹 "{model_path)-gen-images" 中。

## 3. 一致性测试分析 (optional)

为了定量分析推理的一致性，您应确保MS和PT使用相同的初始latent noise和text propmpt，具体操作步骤如下
- 保存在diffusers中使用初始latent noise
    在 `path/to/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`中修改`prepare_latents`函数，加入噪声保存功能。

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

	执行diffusers推理`infer_lora.py`后，初始latent noise将被保存 /tmp/sdxl_init_latents.npy

- 在MindONE中加载噪声npy进行推理

    在如下推理脚本中修改`init_latent_path`和`prompt`，`init_latent_path`指定初始噪声npy文件路径，`prompt`设定对应的文本。

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
    另外需要注意，如果希望尽量跟diffusers保持一致，请将discretization和precision mode设成如上的参数。

## 结果

使用同一个lora checkpoint (在Pokemon数据集上微调)进行推理，MS 和 PT LoRA的生成结果如下。

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/3b664498-f82d-49a9-ad06-876647579d15" width="30%" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/e761ba93-bf97-4bc3-a6d1-4caccdd1614d" width="30%" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/0ef7b3e2-0582-4856-bff5-51b95b9503ee" width="30%" />
</div>
<p align="center">
  <em> MindSpore生成结果（在Pokemon数据集微调后推理） </em>
</p>

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/040c455b-21bd-4bf0-8818-d7378e55d67c" width="30%" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/c72272e7-9757-4667-ae9d-7e1115ddc56d" width="30%" />
<img src="https://github.com/SamitHuang/mindone/assets/8156835/02654e25-ed6b-41dd-830a-bd2b63d04d84" width="30%" />
</div>
<p align="center">
  <em> Torch(diffusers)生成结果(使用相同权重，转换后推理) </em>
</p>

从生成的图像中可以看出，MS 和 PT 生成的图像高度一致。通过定量分析，MS 和 PT 生成的图像之间的平均绝对像素误差小于5。
