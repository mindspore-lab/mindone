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

<details onclose>

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

(Option) Step3. Replace and convert VAE, Download [vae-fp16-fix weights](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) from huggingface.

```shell
python convert_diffusers_to_mindone_sdxl.py \
  --model_path /PATH TO/sdxl-vae-fp16-fix \                 # dir of vae weight
  --vae_name diffusion_pytorch_model.safetensors \          # source vae weight, from https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
  --sdxl_base_ckpt /PATH TO/sd_xl_base_1.0_ms.ckpt          # base checkpoint, from Step2
  --checkpoint_path /PATH TO/sd_xl_base_1.0_vaefix_ms.ckpt  # output path
```

</details>

For details, please refer to [weight_convertion.md](./weight_convertion.md).

### Dataset Preparation for Fine-Tuning (Optional)

#### The text-image dataset for fine-tuning should follow the file structure below

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

> Please run `tools/data_check/get_wds_num_samples.py` on the new dataset to get the sample size of new dataset.

<details onclose>

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

##### WIDS
We provide a dataloader for webdataset (`T2I_Webdataset_RndAcs`) that is compatible with minddata GeneratorDataset.

1. Set the training YAML config as follows to use the T2I_Webdataset loader.
    ```yaml
        dataset_config:
            target: gm.data.dataset_wds.T2I_Webdataset_RndAcs
            params:
                caption_key: 'text_english'
    ```

    A reference config file is shown in `configs/training/sd_xl_base_finetune_910b_wids.yaml`

2. Set `--data_path` in the training script with the path to the data root of the whole training dataset, e.g. `data_dir` in the above example.

Note that the dataloader is implemented based on [wids](https://github.com/webdataset/webdataset?tab=readme-ov-file#the-wids-library-for-indexed-webdatasets). It requires a shardlist information file describing each tar file path and the number of samples in the tar file.

A shardlist decription obeys the following format.
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

You can manually specify a new shardlist description file in the config yaml via the `shardlist_desc` argument, for example.

```yaml
    dataset_config:
        target: gm.data.dataset_wds.T2I_Webdataset_RndAcs
        params:
            caption_key: 'text_english'
            shardlist_desc: 'data_dir/data_info.json'
```

For distributed training, no additional effort is required when using `T2I_Webdataset_RndAcs` dataloader, since it's compatible with mindspore `GeneratorDataset` and the data partition will be finished in `GeneratorDataset` just like training with original data format.


##### Original Webdataset

We also provide a dataloader for original webdataset (`T2I_Webdataset`) that is compatible with minddata GeneratorDataset.

A reference config file is shown in `configs/training/sd_xl_base_finetune_910b_wds.yaml`.

The shardlist description file used here shares the same format as wids.

**Caustion!!** Since we need to know the total number of samples for data parallel training, we provides three ways to get the dataset size of webdataset:

    1. Specify the total number of samples via training config yaml
        ```yaml
        dataset_config:
            target: gm.data.dataset_wds.T2I_Webdataset
            params:
                caption_key: 'text_english'
                num_samples: 10000  # specify total number of samples
        ```
        If `num_samples` is not specify or -1, the following 2 ways will be used to get dataset size.

    2. Get total number of samples from shardlist record
        If shardlist description file is provided in source dataset (see format above), the datsat size will be obtained from the description file. Shardlist description file default path is `{dataset_dir/data_info.json}`.

    3. Scan tar files to record number of samples
        If neither the total number of samples or the shardlist record is provided, we will scanning all tar files to generate the sharlist description file and get the dataset size. It can be time-consuming for larget dataset.


> Note that if you have updated the training data, you should either specify a new shardlist description file or **remove the existing shardlist file** `{data_dir}/data_info.json` for auto re-generation.


</details>

## Inference

### Online Infer

We provide a demo for text-to-image sampling in `demo/sampling_without_streamlit.py` and `demo/sampling.py` with [streamlit](https://streamlit.io/).

After obtaining the weights, place them into checkpoints/. Next, start the demo using

1. (Recommend) Run with interactive visualization:

```shell
# (recommend) run with streamlit
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run demo/sampling.py --server.port <your_port>
```

> If you have network issues on downloading clip tokenizer, please manually download `openai/clip-vit-large-patch14` from huggingface and change `version: openai/clip-vit-large-patch14` in `configs/inference/sd_xl_base.yaml` to `version: your_path/to/clip-vit-large-patch14`

2. Run with other methods:

<details close>

```shell
# run sdxl-base txt2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend

# run sdxl-refiner img2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task img2img \
  --config configs/inference/sd_xl_refiner.yaml \
  --weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --img /PATH TO/img.jpg \
  --device_target Ascend

# run pipeline without streamlit on Ascend
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
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,SDXL-base-1.0_2000_lora.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend
```

</details>

<details>

  <summary>Long Prompts Support</summary>

  By Default, SD-XL only supports the token sequence no longer than 77. For those sequences longer than 77, they will be truncated to 77, which can cause information loss.

  To avoid information loss for long text prompts, we can divide one long tokens sequence (N>77) into several shorter sub-sequences (N<=77) to bypass the constraint of context length of the text encoders. This feature is supported by `args.support_long_prompts` in `demo/sampling_without_streamlit.py`.

  When running inference with `demo/sampling_without_streamlit.py`, you can set the arguments as below.

  ```bash
  python demo/sampling_without_streamlit.py \
  ...  \  # other arguments configurations
  --support_long_prompts True \  # allow long text prompts
  ```

When running inference with `demo/sampling.py`, you can simply input your long prompt and click the button of "Use long text prompt support (token length > 77)" under the prompt, and then start sampling.

</details>

### Offline Infer

See [offline_inference](./offline_inference/README.md).

### Inference with T2i-Adapter

[T2I-Adapter](../t2i_adapter/README.md) is a simple and lightweight network that provides extra visual guidance for
Stable Diffusion models without re-training them. The adapter act as plug-ins to SDXL models, making it easy to
integrate and use.

For more information on inference and training with T2I-Adapters, please refer
to [T2I-Adapter](../t2i_adapter/README.md) page.

### Inference with ControlNet

[ControlNet](https://arxiv.org/abs/2302.05543) controls pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small. Large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

For more information about ControlNet, please refer
to [ControlNet](controlnet.md) page.

### Invisible Watermark Detection

To be supplemented


## Training and Fine-Tuning

⚠️ This function is experimental. The script fine-tunes the whole model and often the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyper-parameters to get the best result on your dataset.

We are providing example training configs in `configs/training`. To launch a training, run

#### 1. Vanilla fine-tune, example as:

<details close>

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

# sdxl-base fine-tune with cache on Ascend
bash scripts/cache_data.sh /path_to/hccl_8p.json 0 8 8 /path_to/dataset/  # cache data
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_8p.json 0 8 8 /path_to/dataset/  # run on server 1

# sdxl-base fine-tune with 16p on Ascend
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_16p.json 0 8 16 /path_to/dataset/  # run on server 1
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_16p.json 8 16 16 /path_to/dataset/ # run on server 2
```

</details>

For details, please refer to [vanilla_finetune.md](./vanilla_finetune.md)

#### 2. LoRA fine-tune, example as:

<details close>

```shell
# sdxl-base lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --gradient_accumulation_steps 4 \
```
</details>

For details, please refer to [lora_finetune.md](./lora_finetune.md)

#### 3. DreamBooth fine-tune

For details, please refer to [dreambooth_finetune.md](./dreambooth_finetune.md).

#### 4. Textual Inversion fine-tune

For details, please refer to [textual_inversion_finetune.md](./textual_inversion_finetune.md).

#### 5. Long prompts support, example as:

<details close>

By default, SDXL only supports the token sequence no longer than 77. For those sequences longer than 77, they will be truncated to 77, which can cause information loss.

To avoid information loss for long text prompts, we add the feature of long prompts training. Long prompts training is supported by `args.lpw` in `train.py`.

```shell
python train.py \
  ...  \  # other arguments configurations
  --lpw True \
```
</details>

#### 6. EDM training

<details close>

> [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364.pdf)

By default, SDXL uses DDPM for training. It can be changed to the EDM-style training by configuring the `denoiser` and other related parameters of the training.

We have provided a EDM-style-training yaml configuration file, in which parameters `denoiser_config` its associated `weighting_config` and `scaling_config` are modified to support EDM training. You can refer to the following case to make it effective.

```shell
python train.py \
  ...  \  # other arguments configurations
  --config configs/training/sd_xl_base_finetune_910b_edm.yaml \
```
</details>
