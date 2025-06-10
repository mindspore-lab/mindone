# Datasets

## Falcon-RefinedWeb

Firstly, please install `datasets` via pip:
```bash
pip install datasets huggingface_hub
pip install -U "huggingface_hub[cli]"
```

Then download Falcon-RefinedWeb from this [URL](https://huggingface.co/datasets/tiiuae/falcon-refinedweb).

You may download the whole dataset from webpage, but we recommand you to download it use:
```bash
from datasets import load_dataset
dataset = load_dataset("tiiuae/falcon-refinedweb", cache_dir="./falcon-refinedweb")
```

If you only need to download a single file from the dataset, you can use:
```bash
from datasets import load_dataset
data_file="data/train-00000-of-*.parquet" # change to your data file
dataset = load_dataset("tiiuae/falcon-refinedweb", cache_dir="./falcon-refinedweb", data_files=data_file)
```
Similarly, you can download this single file via Command Line Interface (CLI):
```bash
huggingface-cli download tiiuae/falcon-refinedweb data/train-00000-of-*.parquet --local-dir ./falcon-refinedweb --repo-type dataset
```

## Laion-Aesthetics-12M

Laion-Aesthetics-12M is available at [Huggingface](https://huggingface.co/datasets/laion/laion-aesthetics-12m).

To download this dataset, please install [`img2dataset`](https://github.com/rom1504/img2dataset) via:

```bash
pip install img2dataset
```

An example tutorial on how to download Laion-Aesthetics dataset can be found [here](https://github.com/rom1504/img2dataset/blob/main/tutorials/laion_aesthetics_12m.md). We recommend you to run:
```bash
mkdir laion-aesthetics-12m && cd laion-aesthetics-12m
wget https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap/resolve/main/train.parquet

cd ..
img2dataset --url_list laion-aesthetics-12m --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion-aesthetics-12m-data --processes_count 16 --thread_count 64 --image_size 384\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
             --save_additional_columns "['similarity','hash','punsafe','pwatermark']"

```

## ImageNet-1K

ImageNet-1K is available at [Huggingface](https://huggingface.co/datasets/imagenet-1k).
