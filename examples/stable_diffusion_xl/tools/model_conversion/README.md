# Model Conversion between Mindspore and Torch Inference

We provide scripts of the checkpoint conversion between `.safetensors`  from Torch and `.ckpt` format from MindSpore. For training or inference in Mindspore, we use the pre-trained weights from hugging face and convert them to `.ckpt` format. Please refer to [weight_convertion](../../docs/preparation.md#convert-pretrained-checkpoint) for details. After finetuning in Mindspore, we can convert the checkpoints back for Torch inference as well.

The tutorial shows how to run inference in the official SDXL repo, [generative-models](https://github.com/Stability-AI/generative-models), with Mindspore checkpoints. Please refer to this [tutorial](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl/tools/lora_conversion/README.md) if running in diffusers.

## Getting started

Notes: if you use Vallina fine-tune, or set `lora_merge_weights` to `true` when using LoRA, you directly get the finetuned weight of the whole model, please skip step 1.

**Step 1. Merge fine-tuned LoRA weights to the pertained base weight**

According to LoRA, $W_{finetuned} = W_{pretrained}+B\times A\times scale$. We can merge our trained LoRA weights with the pre-trained weights by running,

```shell
python merge_lora_to_base.py \
  --weight_lora {path to mindspore lora ckpt} \
  --weight_pretrained ./checkpoints/sd_xl_base_1.0_ms.ckpt \
  --weight_merged ./checkpoints/sd_xl_base_finetuned_ms.ckpt
```

The default path of the merged checkpoint is `./checkpoints/sd_xl_base_finetuned_ms.ckpt`.

**Step 2. Convert ms checkpoint to pt**

To convert the finetuned mindspore checkpoints, run as follows.

```shell
python convert_weight.py \
  --task ms_to_st \
  --weight_safetensors ./checkpoints/sd_xl_base_finetuned_pt.safetensors \
  --weight_ms ./checkpoints/sd_xl_base_finetuned_ms.ckpt  \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml
```

The default path of the converted checkpoint is `./checkpoints/sd_xl_base_finetuned_pt.safetensors`.

**Step 3. Run inference in [generative-models](https://github.com/Stability-AI/generative-models)**

Replace the ckpt path of SDXL-base-1.0 with `sd_xl_base_finetuned_pt.safetensors` at the constant `VERSION2SPECS` in `scripts/demo/sampling.py`, as well as the prompt in ``__main__`.  Then you run inference in generative-models repo with our fine-tuned checkpoint.

```shell
# run in generative-models repo
streamlit run scripts/demo/sampling.py --server.port <your_port>
```

**Step 4. Check consistency between PT and MS inference results (optional)**.

To check inference consistency quantitatively, you should ensure MS and PT use the same initial latent noise and text prompt for diffusion sampling. Here are reference instructions to achieve it.

* Save the initial latent noise used in generative-models

  In `scripts/demo/streamlit_helpers.py` , add 2 lines to the `do_sample` function to save init noise as numpy as follows.

  ```python
  def do_sample():
     ...
     randn = torch.randn(shape).to("cuda")
     # save the init noise as numpy
     import numpy as np
     np.save("/tmp/rand_init_noise.npy", randn.cpu().numpy())
     ...
  ```

  The initial noise will be saved in `/tmp/rand_init_noise.npy`.

* Use the same latent noise in MS inference

  Please set `init_latent_path` and `prompt` in MS inference script referring to the following script.

  ```shell
  finetuned_ckpt_path='./checkpoints/sd_xl_base_finetuned_ms.ckpt'
  init_latent_path='./tmp/rand_init_noise.npy'

  python demo/sampling_without_streamlit.py \
    --task txt2img \
    --config configs/inference/sd_xl_base.yaml \
    --weight $finetuned_ckpt_path \
    --prompt "a sks dog in a dog house" \
    --init_latent_path $init_latent_path \
    --device_target Ascend
  ```

## Results

Here are the generation results for comparison between MS and PT of Dreambooth via LoRA inference, where the Dreambooth-LoRA checkpoint is derived by fine-tuning the [dog](https://github.com/google/dreambooth/tree/main/dataset/dog) dataset.

The generated images for MS (left) and PT (right) are highly consistent as we can see. Quantitatively, the average absolute pixel error between MS and PT-generated images is below 5.

<div align="center">
<img src="https://github.com/mindspore-lab/mindone/assets/33061146/9f4f1ed8-e8d0-447c-9d13-5d1e441920b5" width="80%" />
</div>
<p align="center">
  <em> MindSpore(left) and PyTorch(right) generation results using the Dreambooth via LoRA checkpoint fine-tuned on the dog dataset </em>
</p>

## Convert diffusers pipeline of XL to original stable diffusion

To convert a HF Diffusers saved pipeline to a Stable Diffusion checkpoint with `convert_diffusers_to_original_sdxl.py`, run as follows.

Notes: if you want to save weights in half precision, you can add `--half`. Additionally, you can add `--use_safetensors` to save weights use safetensors. Only converts the UNet, VAE, and Text Encoder.

```shell
cd tools/model_conversion

python convert_diffusers_to_original_sdxl.py \
  --model_path /PATH_TO_THE_MODEL_TO_CONVERT \
  --checkpoint_path /PATH_TO_THE_OUTPUT_MODEL/sd_xl_base_1.0.safetensors \
  --use_safetensors \
  --unet_name "diffusion_pytorch_model.fp16.safetensors" \
  --vae_name "diffusion_pytorch_model.fp16.safetensors" \
  --text_encoder_name "model.fp16.safetensors" \
  --text_encoder_2_name "model.fp16.safetensors"
```

## Convert diffusers pipeline of XL to MindOne stable diffusion

To convert a HF Diffusers saved pipeline to MindOne Stable Diffusion checkpoint with `convert_diffusers_to_mindone_sdxl.py`, run as follows.

Notes: if you want to save weights in half precision, you can add `--half`. Only converts the UNet, VAE, and Text Encoder.

```shell
cd tools/model_conversion

python convert_diffusers_to_mindone_sdxl.py \
  --output_path /PATH_TO_THE_OUTPUT_MODEL/converted_sd_xl_base_1.0.ckpt  \
  --unet_path "diffusion_pytorch_model.fp16.safetensors" \
  --vae_path "diffusion_pytorch_model.fp16.safetensors" \
  --text_encoder_path "model.fp16.safetensors" \
  --text_encoder_2_path "model.fp16.safetensors" \
  --sdxl_base_ckpt "/PATH_TO_THE_BASE_MINDONE_CKPT/sd_xl_base_1.0_ms.ckpt"
```
