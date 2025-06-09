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
