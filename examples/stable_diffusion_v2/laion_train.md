

## Training Stable Diffusion on LAION-art dataset

LAION-art is a 8M laion5B subset with aesthetic > 8, pwatermark < 0.8, punsafe < 0.5. 

We will use it as an example to illustrate how to train a SD model on LAION dataset.

### 1. Download metadata in parquet format

Download the metadata from https://huggingface.co/datasets/laion/laion-art/  

```shell
mkdir laion-art && cd laion-art
wget https://huggingface.co/datasets/laion/laion-art/resolve/main/laion-art.parquet
```

The parquet files contain metadata including image shape, text, language, etc. Here is a data sample read from the parquet file.

```text
{'URL': 'https://www.advocate-art.com/system/ART/Modules/Application/Images/Image/images/000/011/121/artistique_half/VML175107.jpg?6f71107bead7921d03fa7dc3e4ac4b9a8f24dfd3d823d512d7618c06e4059513',
 'TEXT': 'Christmas Shopping Copy',
 'WIDTH': 850,
 'HEIGHT': 850,
 'similarity': 0.2666246294975281,
 'LANGUAGE': 'nolang',
 'hash': -3604776403351267688,
 'pwatermark': 0.0396263524889946,
 'punsafe': 0.00027811527252197266,
 'aesthetic': 8.352225303649902}
```

### 2. Download image data using img2dataset 

Download the image files from URL and organize them into folders using the follow script.

```shell
output_format="files"

img2dataset --url_list /home/yx/datasets/diffusion/laion_art/laion-art.parquet --input_format "parquet" \
        --url_col "URL" --caption_col "TEXT" \
		--output_format $output_format \
        --output_folder laion-art \
		--processes_count 16 --thread_count 64 --image_size 512 \
        --resize_only_if_bigger=True \
		--resize_mode="keep_ratio" \
		--skip_reencode=True \
        --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]' \
		#--enable_wandb True

```


3. Convert to trainable data format 

Get the image-caption txt file for training.


4. Distributed Training




## Reproducing SD 2.1-base by finetuning SD 2.0-base on LAION 5B subsets 

LAION 2B-en 

1. Download parquest files

```shell
mkdir laion_2b_en_ae4.5 && cd laion_2b_en_ae4.5
for i in {1..64}; do wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_4.5plus/resolve/main/2B-en-4.5_55.parquet; done
cd ..
```

2. Download 
