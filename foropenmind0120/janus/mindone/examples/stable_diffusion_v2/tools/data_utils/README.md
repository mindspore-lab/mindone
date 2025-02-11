# LAION Subset Preparation

This doc describes the pipeline for preparing LAION subset for Stable Diffusion training, including metadata download, filtering, source image downloading, and annotation generation.

## Introduction
[LAION](https://laion.ai/blog/laion-5b/) is a large-scale image-text dataset with over 5 billion samples. We will pick a subset of LAION-5B for SD-2.1-base training, where the subset filtering conditions are as follows.

```text
    lang=en
    aesthetic score>=4.5
    punsafe <= 0.98
    resolution >= 512x512
```

Since we only aim at English text-image generation, we can just use [LAION2b-en](https://huggingface.co/datasets/laion/laion2B-en) for data selection. Note the source dataset only contains metadata (stored as `.parquet` files), each sample includes text, image URL, safety score, aesthetic score, etc. No images are stored in the metadata and we will download them from the provided image URLs.

For simplicity, you can directly download the metadata filtered for SD2.1-base training from [this link](https://huggingface.co/datasets/jasonhuang23/laion2b_en_sd2.1base) using the following script
```shell
mkdir laion_metadata && cd laion_metadata
for i in {1..64}; do wget https://huggingface.co/datasets/jasonhuang23/laion2b_en_sd2.1base/resolve/main/part_{i}.parquet; done
cd ..
```

If you want to filter the metadata by other conditions, please refer to Step 1 and 2 below. Otherwise, you can skip to Step 3 after installing the dependency and downloading the prepared metadata.

## Dependency Installation

We will use `pyspark` to do metadata filtering and `img2dataset` to download source images. Please install the required packages by:

For Linux:
```shell
apt-get install openjdk-8-jdk
pip install pyspark

pip install img2dataset
```

For MacOS:
```shell
brew install openjdk@11
brew install apache-spark
pip install pyspark
pip install img2dataset
```

For Windows:
```shell
pip install pyspark
pip install img2dataset
```

To reduce image download failures caused by certificate verify issue, you may do the following code patching:

```shell
mkdir tmp; cd tmp
git clone https://github.com/SamitHuang/mindone.git --branch laion_p1
patch_fp=mindone/examples/stable_diffusion_v2/tools/laion_data_utils/img2dataset_patch/downloader.py
ori_downloader=$(python -c 'import img2dataset; print(img2dataset.downloader.__file__)' | awk '{print $1}')
mv $patch_fp $ori_downloader
cd ..; rm -rf tmp
echo Finish updating img2dataset $ori_downloader
```
## Step 1. Download LAION 2B-en Metadata

Execute the following commands in the terminal to download the whole laion 2b-en metadata.

```shell
mkdir laion_2b_en_ae4.5 && cd laion_2b_en_ae4.5
for i in {1..64}; do wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_4.5plus/resolve/main/2B-en-4.5_$i.parquet; done
cd ..
```
> Use `wget --no-check-certificate {URL}` instead if no-certifcate-verified error occurs.

It results in 64 parquet files with 1,372,052,778 samples in total, which takes **214GB**.

An example sample is as follows:
```text
{'URL': 'https://img1.etsystatic.com/186/0/5531524/il_340x270.1310915051_5kax.jpg',
 'TEXT': 'Vintage Green Glass Swag Hanging Lamp Mid Century',
 'WIDTH': 340.0,
 'HEIGHT': 270.0,
 'similarity': 0.3491742014884949,
 'punsafe': 8.991600225272123e-06,
 'pwatermark': 0.14151343703269958,
 'AESTHETIC_SCORE': 4.751741409301758,
 'hash': 6170073934346815248,
 '__index_level_0__': 0}
```

Note that the key names can vary for different LAION subsets. For laion_art dataset, they are
```text
'URL', 'TEXT', 'WIDTH', 'HEIGHT', 'similarity', 'LANGUAGE', 'hash', 'pwatermark', 'punsafe', 'aesthetic']
```

## Step 2. Filter the Metadata

We will exclude the samples in metadata that are not needed for training, which can save downloading time and storage space.

```shell
python laion_filter_metadata.py --data_dir laion_2b_en_ae4.5  --output_dir laion_2b_en_filtered
```

where the filter conditions used in the script are as follows (you may change them in `laion_filter_metadata.py`)
```text
WIDTH>=512
HEIGHT>=512
punsafe<=0.98
AESTHETIC_SCORE>=4.5
 ```

It results in **64 parquet files** with 340,954,340 samples in total, which takes **56GB**.

For convenience, the filtered metadata is uploaded to [here](https://huggingface.co/datasets/jasonhuang23/laion2b_en_sd2.1base). You may download them directly used for source image downloading.


## Step 3. Download Source Images and Resize

In this step, we will use `img2dataset` to download source image files, and resize, and encode them into target format.

**Notes**: The overall dataset will take up about **30TB**, which is beyond the capacity of most hard drives. So you should either

1) download different parts to different hard drives locally at first, then upload them to the storage server (e.g. OBS) before the hard drive reaches its capacity limit,

or 2) directly download to the storage server with local caching if its remote file system supports it. The optimal choice depends on your network condition and features of the remote file system.

### Download to Local Drives

Please modify the `laion_download_imgs.sh` script by setting `metadata_dir` to the directory of downloaded metadata, setting `output_dir` to where you want to save the donwloaded source images, for example.

```shell
metadata_dir="/MyDisk/laion2b_en/sd2.1_base_train/metadata_filtered" # change to your local path containing the filtered metadata
output_folder="/MyDisk/laion2b_en/sd2.1_base_train/image_text_data" # change to your local path for saving the downloaded images
```
To download one part of the dataset, run
```shell
bash laion_download_imgs.sh {part_id}`, e.g. `bash laion_download_imgs.sh 1`
```

To download multiple parts in sequence (e.g., part 1, 2, 3), run
``` shell
for part_id in 1 2 3; do bash laion_download_imgs.sh $part_id; done
```

#### Hints on Improving Download Performance:
- Set `processes_count` as the number of CPU cores your machine has, and increase `thread_count` as long as the bandwidth and CPU are below limit (100%).
- Some urls can become invalid. The success rate has dropped to around 80% from the day when LAION dataset was released. In CN, the success rate can further drop to around 50%.
- Detailed failure reasons can be checked in the log file in `{save_dir}/part_id/{id}_stats.json` for further fix (certifcate issue can be fixed with the patch introduced in the Installation section)


It will take about 20 hours to download one part with one node, resulting in
``` texts
532 subfolders, each supposed to contain 10,000 images
URL download success rate: ~80% (by 20 July 2023, may fail due to invalid URL or network connection)
Actually downloaded images in each subfolder: ~8,000 images
Total images actually donwloaded for part 1: 4.26M images (4261204)
Total size for part 1 (output_format=files): 459GB
```

There are 64 parts in total, so they will result in ~272M images and take ~30TB for `files` saving format or ~10TB in for `webdataset` format.

### Download to Remote Cluster Directly

If you have a cluster or have access to N machines over ssh, you can set up a PySpark cluster and then download the images directly to the cluster.

For detailed instructions, please refer to [distributed_img2dataset_tutorial](https://github.com/rom1504/img2dataset/blob/main/examples/distributed_img2dataset_tutorial.md) and [laion5B download images](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md#download-the-images).



#### Notes on data format
You can alos change the parameters for image resizing, saving format, etc. Please look into the `laion_download_imgs.sh` script and refer to `img2dataset` [API doc](https://github.com/rom1504/img2dataset/tree/main#api).

To change the saving format, please change the `output_format` arg in the `laion_download_imgs.sh` script. We use `webdataset` format by default (data shard saved as tar file) because it is more convenient to handle packed data rather than billions of image files, and can save space for HDD devices and support suqential access with [WebDataset](https://github.com/webdataset/webdataset).

```shell
#output_format="files"
#output_format="parquet"
output_format="webdataset"
```

## Step 4. Generate Annotation File


### For training with original files (jpg)

If you set `--dataset_type=files` in SD training, please generate annotation files as follows.


```shell
python laion_gen_csv_annot.py --data_dir {path/to/download_folder}
```
> e.g.  `python laion_gen_csv_annot.py --data_dir /data/laion_2b_en`

This is to collect the image paths and their corresponding captions into csv files, used for creating sample indexes in traning data loading.

After execution, the ready-to-train data should be in the following structure.
```text
data_dir
├── part_1.csv # annotation
├── part_1/
│   ├── 00000.csv
│   ├── 00000  # (00000.tar for webdataset)
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   ├── 000000002.jpg
│   │   └── ...
│   ├── ...
│     
├── part_2.csv
├── part_2/
...
```

### For training with webdataset format (tar files)
> Data loading is ~20% faster than raw format and requires much less time in init on SSD.

If you set `--dataset_type=webdataset` in SD training, please generate annotation files as follows.

```shell
python laion_gen_data_stats.py --data_dir {path/to/download_folder}
```
> e.g.  `python laion_gen_data_stats.py --data_dir /data/laion_2b_en`

This is to collect the tar file paths and the number of samples for each tar, which is required for IterableDataset.

After execution, the ready-to-train data should be in the following structure.
```text
data_dir
├── part_1_stats.csv # annotation, record: tar file path, number of samples
├── part_1/
│   ├── 00000.tar ( archive of image files and the corresponding text and metadata as follows)
│   │   ├── 000000000.jpg
│   │   ├── 000000000.json
│   │   ├── 000000000.text
│   │   ├── 000000001.jpg
│   │   ├── 000000001.json
│   │   ├── 000000001.text
│   │   └── ...
│   ├── ...
│     
├── part_2_stats.csv
├── part_2/
...
```
