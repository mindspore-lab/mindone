# Preparation

## Requirements

| mindspore      | ascend driver | firmware    | cann toolkit/kernel |
|:--------------|:-------------|:-----------|:------------------|
| 2.3.1    | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1        |


To install other dependencies, please run

```shell
pip install -r requirements.txt
```

## Convert Pretrained Checkpoint

We provide a script for converting pre-trained weight from `.safetensors` to `.ckpt` in `tools/model_conversion/convert_weight.py`.

step1. Download the [Official](https://github.com/Stability-AI/generative-models) pre-train weights [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) from huggingface.

step2. Convert weight to MindSpore `.ckpt` format and put it to `./checkpoints/`.

```shell
cd tools/model_conversion

# convert sdxl-base-1.0 model
python convert_weight.py \
  --task st_to_ms \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml

# convert sdxl-refiner-1.0 model
python convert_weight.py \
  --task st_to_ms \
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


## Datasets Preparation for Fine-Tuning (Optional)

Please also refer to the docs of specific finetune methods for data preparation.

### General text-image datasets

In general, the text-image dataset for fine-tuning should follow the file structure below,

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

We prepare two public text-image datasets obeying the above format for convenience.

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.
- [Chinese-art blip caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 100 Chinese art-style images with BLIP-generated captions.

Please download `pokemon_blip.zip` or `chinese_art_blip.zip` from the [openi dataset website](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets) and then unzip.


### Training with Webdataset

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

> Tools: run `tools/data_check/get_wds_num_samples.py` on the new dataset to get the sample size of new dataset.

#### WIDS
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

The dataloader is implemented based on [wids](https://github.com/webdataset/webdataset?tab=readme-ov-file#the-wids-library-for-indexed-webdatasets). It requires a shardlist information file describing each tar file path and the number of samples in the tar file.

A shardlist description obeys the following format.
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

You can manually specify a new shardlist description file in the config yaml via the `shardlist_desc` argument, for example,

```yaml
    dataset_config:
        target: gm.data.dataset_wds.T2I_Webdataset_RndAcs
        params:
            caption_key: 'text_english'
            shardlist_desc: 'data_dir/data_info.json'
```

For distributed training, no additional effort is required when using `T2I_Webdataset_RndAcs` dataloader. It's compatible with mindspore `GeneratorDataset,` and the data partition will be finished in `GeneratorDataset` just like training with the original data format.


#### Original Webdataset

We also provide a dataloader for the original webdataset (`T2I_Webdataset`) that is compatible with minddata GeneratorDataset.

A reference config file is shown in `configs/training/sd_xl_base_finetune_910b_wds.yaml`.

The shardlist description file used here shares the same format as wids.

**Caution!!** Since we need to know the total number of samples for data parallel training, we provide three ways to get the dataset size of webdataset:

    1. Specify the total number of samples via training config yaml
        ```yaml
        dataset_config:
            target: gm.data.dataset_wds.T2I_Webdataset
            params:
                caption_key: 'text_english'
                num_samples: 10000  # specify the total number of samples
        ```
        If `num_samples` is not specified or -1, the following 2 ways will be used to get dataset size.

    2. Get total number of samples from shardlist record
        If shardlist description file is provided in source dataset (see format above), the datasat size will be obtained from the description file. Shardlist description file default path is `{dataset_dir/data_info.json}`.

    3. Scan tar files to record number of samples
        If neither the total number of samples or the shardlist record is provided, we will scanning all tar files to generate the sharlist description file and get the dataset size. It can be time-consuming for a large dataset.


If you have updated the training data, please either specify a new shardlist description file or **remove the existing shardlist file** `{data_dir}/data_info.json` for auto re-generation.
