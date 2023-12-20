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
export MS_PYNATIVE_GE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run demo/sampling.py --server.port <your_port>
```

- Run with other methods:

```shell
# run sdxl-base txt2img without streamlit on Ascend
export MS_PYNATIVE_GE=1
python demo/sampling_without_streamlit.py \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
```

### 2. Inference with SDXL-Refiner

```shell
# run sdxl-refiner img2img without streamlit on Ascend
export MS_PYNATIVE_GE=1
python demo/sampling_without_streamlit.py \
  --task img2img \
  --config configs/inference/sd_xl_refiner.yaml \
  --weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --img /PATH TO/img.jpg \

# run pipeline without streamlit on Ascend
export MS_PYNATIVE_GE=1
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --add_pipeline True \
  --pipeline_config configs/inference/sd_xl_refiner.yaml \
  --pipeline_weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
```

### 3. Inference with T2i-Adapter

[T2I-Adapter](../t2i_adapter/README.md) is a simple and lightweight network that provides extra visual guidance for
Stable Diffusion models without re-training them. The adapter act as plug-ins to SDXL models, making it easy to
integrate and use.

For more information on inference and training with T2I-Adapters, please refer
to [T2I-Adapter](../t2i_adapter/README.md) page.

### 4. Support List

<div align="center">

| Model Name    | Device      | MindSpore        | CANN | ImageSize | Compile Cost | Sampler  | Sample Step | Step Time | Sample Time |
|---------------|-------------|------------------|------|-----------|--------------|----------|-------------|-----------|-------------|
| SDXL-Base     | Ascend 910* | mindspore 2.2.10 | C15  | 1024x1024 | 302s         | EulerEDM | 40          | Testing   | 8.1s        |
| SDXL-Refiner  | Ascend 910* | mindspore 2.2.10 | C15  | 1024x1024 | Testing      | EulerEDM | 40          | Testing   | Testing     |
| SDXL-PipeLine | Ascend 910* | mindspore 2.2.10 | C15  | 1024x1024 | Testing      | EulerEDM | 35/5        | Testing   | Testing     |

</div>
<br>

## Offline Inference

See [offline_inference](./offline_inference/README.md).

## Invisible Watermark Detection

To be supplemented
