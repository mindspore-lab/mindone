# Inference

<img src="https://github.com/mindspore-lab/mindone/assets/20476835/68d132e1-a954-418d-8cb8-5be4d8162342" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/9f0d0d2a-2ff5-4c9b-a0d0-1c744762ee92" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/dbaf0c77-d8d3-4457-b03c-82c3e4c1ba1d" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/f52168ef-53aa-4ee9-9f17-6889f10e0afb" width="240" />

> Note: sampled 40 steps by SDXL-1.0-Base on Ascend 910 (online inference).

## Online Inference

We provide a demo for text-to-image sampling in `demo/sampling_without_streamlit.py` and `demo/sampling.py` with [streamlit](https://streamlit.io/).

After obtaining the weights, place them into `checkpoints/`. Next, start the demo using

> Note: If you have network issues on downloading clip tokenizer, please manually download [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) from huggingface and change `version: openai/clip-vit-large-patch14` in `configs/inference/sd_xl_base.yaml` to `version: your_path/to/clip-vit-large-patch14`

### Dependency

- mindspore 2.2.10

To install the dependency, please run

```shell
pip install -r requirements.txt
```

### Pretrained models

Download the official pre-train weights from huggingface, convert the weights from `.safetensors` format to Mindspore `.ckpt` format, and put them to `./checkpoints/` folder. Please refer to SDXL [weight_convertion.md](./weight_convertion.md) for detailed steps.

### 1. Inference with SDXL-Base

- (Recommend) Run with interactive visualization:

```shell
# (recommend) run with streamlit
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run demo/sampling.py --server.port <your_port>
```

- Run with other methods:

```shell
# run sdxl-base txt2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
```

### 2. Inference with SDXL-Refiner

```shell
# run sdxl-refiner img2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task img2img \
  --config configs/inference/sd_xl_refiner.yaml \
  --weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --img /PATH TO/img.jpg

# run pipeline without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --add_pipeline True \
  --pipeline_config configs/inference/sd_xl_refiner.yaml \
  --pipeline_weight checkpoints/sd_xl_refiner_1.0_ms.ckpt
```

### 3. Inference with T2i-Adapter

[T2I-Adapter](../t2i_adapter/README.md) is a simple and lightweight network that provides extra visual guidance for
Stable Diffusion models without re-training them. The adapter act as plug-ins to SDXL models, making it easy to
integrate and use.

For more information on inference and training with T2I-Adapters, please refer
to [T2I-Adapter](../t2i_adapter/README.md) page.

### 4. Inference with ControlNet

[ControlNet](https://arxiv.org/abs/2302.05543) controls pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small. Large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

For more information about ControlNet, please refer to [ControlNet](controlnet.md) page.

### 5. Inference with different schedulers

A scheduler defines how to iteratively add noise to an image in training and how to update a sample based on a model’s output in inference.

SDXL uses the DDPM formulation by default, which is set in `denoiser_config`  in yaml file. See `onfigs/inference/sd_xl_base.yaml`.

[EDM formulation](https://arxiv.org/abs/2006.11239) is supported as well. An example yaml config is `configs/inference/sd_xl_base_edm_pg2_5.yaml`,  where the `weighting_config`,  `scaling_config,` and `discretization_config`  in `denoiser_config`  are modified to `EDMWeighting`,  `EDMScaling` and `EDMDiscretization`.

The `denoiser_config` of the model in yaml config file together with the args of samplers such as `sampler`, `guider` and `discretization` in sampling script define a scheduler in inference. Examples of EDM-style inference are as below.

* EDM formulation of Euler sampler (EDMEulerScheduler)

  ```shell
  python demo/sampling_without_streamlit.py \
    --config configs/inference/sd_xl_base_edm_pg2_5.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
    --sampler EulerEDMSampler \
    --sample_step 20 \
    --guider VanillaCFG  \
    --guidance_scale 3.0 \
    --discretization EDMDiscretization \
    --sigma_min 0.002 \
    --sigma_max 80.0 \
    --rho 7.0
  ```

* EDM formulation of DPM++ 2M sampler (EDMDPMsolverMultistepScheduler)

  ```shell
  python demo/sampling_without_streamlit.py \
    --config configs/inference/sd_xl_base_edm_pg2_5.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
    --sampler DPMPP2MSampler \
    --sample_step 20 \
    --guider VanillaCFG \
    --guidance_scale 5.0 \
    --discretization EDMDiscretization \
    --sigma_min 0.002 \
    --sigma_max 80.0 \
    --rho 7.0
  ```

### 6. Support List

<div align="center">

| Model Name    | Device      | MindSpore        | CANN | ImageSize | Compile Cost |Flash Attention| Sampler  | Sample Step | Sample Time |
|---------------|-------------|------------------|------|-----------|--------------|---------------|----------|-------------|-------------|
| SDXL-Base     | Ascend 910* | mindspore 2.2.12 | 7.0.0 beta1  | 1024x1024 | 182s         | ON            | EulerEDM | 40          | 6.66s       |
| SDXL-Base     | Ascend 910* | mindspore 2.2.12 | 7.0.0 beta1  | 1024x1024 | 182s         | ON            | DPM++2M Karras | 20    | 4.3s        |
| SDXL-Refiner  | Ascend 910* | mindspore 2.2.12 | 7.0.0 beta1  | 1024x1024 | Testing      | ON            | EulerEDM | 40          | Testing     |
| SDXL-PipeLine | Ascend 910* | mindspore 2.2.12 | 7.0.0 beta1  | 1024x1024 | Testing      | ON            | EulerEDM | 35/5        | Testing     |
| SDXL-Base     | Ascend 910  | mindspore 2.2.12 | 7.0.0 beta1  | 1024x1024 | 295s         | OFF           | DPM++2M Karras | 20    | 17s         |
| SDXL-Base     | Ascend 910  | mindspore 2.2.12 | 7.0.0 beta1  | 1024x1024 | 280s         | ON            | DPM++2M Karras | 20    | 14.5s       |
</div>
<br>

Note: Please refer to [FAQ](./faq_cn.md) Question 6 if using Flash Attention on Ascend 910.


## Offline Inference

See [offline_inference](./offline_inference/README.md).


## Invisible Watermark Detection

To be supplemented
