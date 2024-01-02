# Getting Started with SDXL

This document provides a brief introduction to the usage of built-in command-line tools in SDXL.

## Dependency

- mindspore 2.2
- openmpi 4.0.3 (for distributed mode)

To install the dependency, please run

```shell
pip install -r requirements.txt
```

## Preparation

### Convert Pretrained Checkpoint

We provide a script for converting pre-trained weight from `.safetensors` to `.ckpt` in `tools/model_conversion/convert_weight.py`.

step1. Download the [Official](https://github.com/Stability-AI/generative-models) pre-train weights [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) from huggingface.

step2. Convert weight to MindSpore `.ckpt` format and put it to `./checkpoints/`.

```shell
cd tools/model_conversion

# convert sdxl-base-1.0 model
python convert_weight.py \
  --task pt_to_ms \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml

# convert sdxl-refiner-1.0 model
python convert_weight.py \
  --task pt_to_ms \
  --weight_safetensors /PATH TO/sd_xl_refiner_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_refiner_1.0_ms.ckpt \
  --key_torch torch_key_refiner.yaml \
  --key_ms mindspore_key_refiner.yaml
```

### Dataset Preparation for Fine-Tuning (Optional)

The text-image dataset for fine-tuning should follow the file structure below

<details onclose>

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

img_txt.csv is the annotation file in the following format
```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

For convenience, we have prepared two public text-image datasets obeying the above format.

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.
- [Chinese-art blip caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 100 Chinese art-style images with BLIP-generated captions.

To use them, please download `pokemon_blip.zip` or `chinese_art_blip.zip` from the [openi dataset website](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets). Then unzip.

</details>


#### Training with Webdataset

Image-text pair data are archived into `tar` files in webdataset. A training dataset is like
```text
data_dir
├── 00001.tar
│   ├── 00001.jpg
│   ├── 00001.json
│   ├── 00002.jpg
│   ├── 00002.json
│   └── ...
├── 00002.tar
├── 00003.tar
└── ...
```

We provide a dataloader for webdataset (`T2I_Webdataset_RndAcs`) that is compatible with minddata GeneratorDataset.

1. Set the training YAML config as follows to use the T2I_Webdataset loader.
    ```yaml
        dataset_config:
            target: gm.data.dataset_wds.T2I_Webdataset_RndAcs
            params:
                caption_key: 'text_english'
    ```

2. Set `--data_path` in the training script with the path to the data root of the whole training dataset, e.g. `data_dir` in the above example.

Note that the dataloader is implemented based on [wids](https://github.com/webdataset/webdataset?tab=readme-ov-file#the-wids-library-for-indexed-webdatasets), which requires shardlist information which describes the path to each tar file and the number of data samples in the tar file.

For the first time running, the data loader will scan the whole dataset to get the shardlist information (which can be time-consuming for large dataset) and save it as a json file like follows.

```json
{
"__kind__": "wids-shard-index-v1",
"wids_version": 1,
"shardlist":
    [
        {"url": "data_dir/part01/00001.tar", "nsamples": 10000},
        {"url": "data_dir/part01/00002.tar", "nsamples": 10000},
    ]
}
```

To save the time of scanning all data, you should prepare a data description json file ahead following the above format (recording num of samples for each tar file in `nsamples`).  Then parse the prepared json file to the loader via the `shardlist_desc` argument, such as

```yaml
    dataset_config:
        target: gm.data.dataset_wds.T2I_Webdataset_RndAcs
        params:
            caption_key: 'text_english'
            shardlist_desc: 'data_dir/data_info.json'
```

For distributed training, no additional effort is required when using `T2I_Webdataset_RndAcs` dataloader, since it's compatible with mindspore `GeneratorDataset` and the data partition will be finished in `GeneratorDataset` just like training with original data format.

## Inference

### Online Infer

We provide a demo for text-to-image sampling in `demo/sampling_without_streamlit.py` and `demo/sampling.py` with [streamlit](https://streamlit.io/).

After obtaining the weights, place them into checkpoints/. Next, start the demo using

1. (Recommend) Run with interactive visualization:

```shell
# (recommend) run with streamlit
export MS_PYNATIVE_GE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run demo/sampling.py --server.port <your_port>
```

> If you have network issues on downloading clip tokenizer, please manually download `openai/clip-vit-large-patch14` from huggingface and change `version: openai/clip-vit-large-patch14` in `configs/inference/sd_xl_base.yaml` to `version: your_path/to/clip-vit-large-patch14`

2. Run with other methods:

<details close>

```shell
# run sdxl-base txt2img without streamlit on Ascend
export MS_PYNATIVE_GE=1
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend

# run sdxl-refiner img2img without streamlit on Ascend
export MS_PYNATIVE_GE=1
python demo/sampling_without_streamlit.py \
  --task img2img \
  --config configs/inference/sd_xl_refiner.yaml \
  --weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --img /PATH TO/img.jpg \
  --device_target Ascend

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
  --sd_xl_base_ratios "1.0_768" \
  --device_target Ascend

# run lora(unmerge weight) without streamlit on Ascend
export MS_PYNATIVE_GE=1
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,SDXL-base-1.0_2000_lora.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend
```

</details>

### Offline Infer

See [offline_inference](./offline_inference/README.md).

### Inference with T2i-Adapter

[T2I-Adapter](../t2i_adapter/README.md) is a simple and lightweight network that provides extra visual guidance for
Stable Diffusion models without re-training them. The adapter act as plug-ins to SDXL models, making it easy to
integrate and use.

For more information on inference and training with T2I-Adapters, please refer
to [T2I-Adapter](../t2i_adapter/README.md) page.

### Invisible Watermark Detection

To be supplemented


## Training and Fine-Tuning

⚠️ This function is experimental. The script fine-tunes the whole model and often the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyper-parameters to get the best result on your dataset.

We are providing example training configs in `configs/training`. To launch a training, run

1. Vanilla fine-tune, example as:

```shell
# sdxl-base fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \

# sdxl-base fine-tune with 8p on Ascend
mpirun --allow-run-as-root -n 8 python train.py \
  --config configs/training/sd_xl_base_finetune_multi_graph_910b.yaml \
  --weight "" \
  --data_path /PATH TO/YOUR DATASET/ \
  --max_device_memory "59GB" \
  --param_fp16 True \
  --is_parallel True

# sdxl-base fine-tune with 16p on Ascend
bash scripts/run_vanilla_ft_910b_16p /path_to/hccl_16p.json 0 8 16 /path_to/dataset/  # run on server 1
bash scripts/run_vanilla_ft_910b_16p /path_to/hccl_16p.json 8 16 16 /path_to/dataset/ # run on server 2
```

2. LoRA fine-tune, example as:

```shell
# sdxl-base lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --gradient_accumulation_steps 4 \
```

3. DreamBooth fine-tune

For details, please refer to [dreambooth_finetune.md](./dreambooth_finetune.md).

4. Run with Multiple NPUs, example as:

```shell
# run with multiple NPU/GPUs
mpirun --allow-run-as-root -n 8 python train.py \
  --config /PATH TO/config.yaml \
  --weight /PATH TO/weight.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --is_parallel True \
  --device_target <YOUR DEVCIE>
```
