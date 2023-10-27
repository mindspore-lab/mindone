#!/bin/bash

# command: bash download_webvid.sh 10M_train part0 data1

# wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_val.csv

video2dataset --url_list="/data1/webvid-10m/metadata/results_$1/$2.csv" \
        --input_format="csv" \
        --output-format="webdataset" \
	      --output_folder="/$3/webvid-10m/dataset/$1/$2" \
        --url_col="contentUrl" \
        --caption_col="name" \
	      --save_additional_columns='[videoid,page_dir,duration]' \
        #--save_additional_columns='[videoid,page_idx,page_dir,duration]' \
        #--enable_wandb=False \
	      #--config=default \
