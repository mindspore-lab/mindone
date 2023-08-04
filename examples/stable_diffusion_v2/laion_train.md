# Training SD 2.x on LAION datasets

## Data Prepration

### Dependency

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

### Data Description
We will use the following data source and filtering conditions for preparing data for SD 2.1 base training.

- Source metadata - LAION2b-en: https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_4.5plus 
> Compared to the original LAION2b-en with 345GB metadata, this source has about 230GB metadata by fitlering samples with aesthetic score < 4.5.
- Filtering conditions that will be applied in Step 2:
```text
    lang=en
    aesthetic score>=4.5
    punsafe <= 0.98
    resolution >= 512x512
```
- Filtered metadata - LAION2b-en-sd2.1base: https://huggingface.co/datasets/jasonhuang23/laion2b_en_sd2.1base 

### Step 1. Download LAION 2B-en Metadata

#### Saving to Local

Execute the following commands in terminal to download the whole laion 2b-en metadata.

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

#### Saving to Cloud (not tested)

Referring the script of [saving to aws](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md#saving-to-aws): 
```shell
bucket=laion2b_en_aesthetics_4.5plus # bucket in s3 
for i in {1..64}; 
do 
    wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_4.5plus/resolve/main/2B-en-4.5_$i.parquet -O - | aws s3 cp - s3://$bucket/metadata/part_$i.parquet;
done
```
, you can modify it for your own cloud storage platform if it supports data transfer via command line. 

For modelarts obs, the following script should work.
```shell
bucket=laion2b_en_aesthetics_4.5plus # bucket in s3 
cloud_platform_tool=aicc_obs  # this is an example for cloud storage command tool 
for i in {1..64}; 
do 
    wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_4.5plus/resolve/main/2B-en-4.5_$i.parquet -O - | $cloud_platform_tool s3 cp - s3://$bucket/metadata/part_$i.parquet;
done
```


### Step 2. Filter the Metadata  

We will exclude the samples in metadata that are not needed for training, which can save downloading time and storage space. 


```shell
python laion_filter_metadata.py --data_dir laion_2b_en_ae4.5  --output_dir laion_2b_en_filtered
```

where the filter conditions are hard-coded in the script as:
```text
WIDTH>=512
HEIGHT>=512
punsafe<=0.98
AESTHETIC_SCORE>=4.5
 ```

It results in **64 parquet files** with 340,954,340 samples in total, which takes **56GB**.

(For convenience, the filtered metadata is also uploaded in [this link](https://huggingface.co/datasets/jasonhuang23/laion2b_en_sd2.1base). You may download it to local or cloud referring to the download script in Step 1.)

### Step 3. Download Source Images and Resize 

In this step, we will use `img2dataset` to download the image files from URLs in the filtered metadata, and resize, encode them into target format.

#### Option 1: Download by Part to Local Drives

If you prefer to save the data locally but the storage divice is small (<=10TB), please take this option.

Firstly, modify `input_folder` and `output_folder` in the `laion_download_imgs.sh` script as follows.
```shell
input_folder="/MyDisk/laion2b_en/sd2.1_base_train/metadata_filtered" # change to your local path containing the filtered metadata
output_folder="/MyDisk/laion2b_en/sd2.1_base_train/image_text_data" # change to your local path for saving the downloaded images
```

Then, run

```shell
# download a part of the whole dataset, where part_id can be an integer in [1, 64].
bash laion_download_imgs.sh {part_id}
```
> Tips: To get best throughput, you should set `processes_count` as the number of CPU cores your machine has, increase `thread_count` as long as the bandwitdh and CPU is below limit (100%).  

, where you need to set `{part_id}` to an integer  in [1, 64], since we will download the large-scale image dataset part by part (64 parts in total). e.g. `bash laion_download_imgs.sh 1`

It will take about 20 hours to download one part with one node and will result in
``` texts
532 subfolders, each supposed to contain 10,000 images
URL download success rate: ~80% (by 20 July 2023)
Actually downloaded images in each subfolder: ~8,000 images
Total images actually donwloaded for part 1: 4.26M images (4261204)
Total size for part 1 (output_format=files): 459GB
```

There are 64 parts in total, so they will result in ~272M images and take ~30TB for `files` saving format or ~10TB in for `webdataset` format.

#### Option 2: Download at Whole to Large Local Drives 

If you have enough storage space >= 32TB, you can download all images at a time by

```shell
# download the whole dataset
bash laion_download_imgs.sh 0
```

#### Option 3: Distributed Downloading with a PySpark Cluster

If you have a cluster or have access to N machines over ssh, you can set up a PySpark cluster then download the images in a distributed way.

For detailed instructions, please refer to [distributed_img2dataset_tutorial](https://github.com/rom1504/img2dataset/blob/main/examples/distributed_img2dataset_tutorial.md) and [laion5B download images](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md#download-the-images).  

> TODO: The given two tutorials provide instructions for aws/aliyun instances, but not for aicc/modelarts. Investigate on how to depoly pyspark on modelarts or ascend servers. 

#### Notes on data storage format
You can alos change the parameters for image resizing, saving format, etc. Please look into the `laion_download_imgs.sh` script and refer to `img2dataset` [API doc](https://github.com/rom1504/img2dataset/tree/main#api).

To change the saving format, please change the `output_format` arg in the `laion_download_imgs.sh` script. We use `webdataset` format by default for its metrits in less storage size (**4+ times** smaller than `files` format on HDD disks) and fast loading speed for sequential access. 

```shell
#output_format="files"
#output_format="parquet"
output_format="webdataset"
```

#### Notes on download failure issue:
- Some urls can become invalid. The success rate has dropped to around 80% from the day when LAION dataset was released,
- Without proxy in CN, the success rate can further drop to around 50%.
- For detailed reasons for download failures, you can check the log file in `{output_dir}/{id}_stats.json` and try to fix them 
    - no-certifate error - fix by adding `-k` to curl download command in img2dataset source code


### Step 4. Generate Annotation File for Training 

#### For training with original files

This step is to record the image paths and their corresponding captions into csv files, used for data indexing in training.

```shell
python laion_gen_csv_annot.py --data_dir {path/to/download_folder}
```
> e.g.  `python laion_gen_csv_annot.py --data_dir /data/laion_2b_en`


After execution, the ready-to-train data should be in following structure.
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


#### For training with tar files (webdataset)
> Data loading is ~20% faster than raw format and requires much less time in init on SSD. 


```shell
python laion_gen_data_stats.py --data_dir {path/to/download_folder}
```
> e.g.  `python laion_gen_data_stats.py --data_dir /data/laion_2b_en`

After execution, the ready-to-train data should be in following structure.
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



## Training

### Training on Ascend Servers

After the data preparation, set up the `data_path` in `scripts/run_train_v2_distributed.sh`

Generate the hccl rank table file referring to [this doc](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/tutorials/distribute_train.md#12-configure-rank_table_file-for-training), then modify parallel config including device num in `scripts/run_train_v2_laion.sh` accordingly. 

To launch distributed training, run

```
sh scripts/run_train_v2_laion.sh
```

Note: Large global batch size is preferred for better model convergence. **2048** is a reference for producing good training results. An example setting to reach it: 64 NPUs with batch_size=3 and accumulation_steps=10.


### Training on AICC or ModelArts (coming soon)

#### 1. Upload the data dir to OBS (skip it if option 3, distributed download on AICC, in Data Preparation Step 3 is achieved.)

#### 2. Set up training job on the webpage

Under development. Please seek documents from AICC providers.  

