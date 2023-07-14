output_format="files"
#output_format="parquet"
input_folder=/data3/datasets/laion_art_metadata_filtered 
output_folder=/data3/datasets/laion_art_filtered # make sure this folder is set on the disk with large enough space (> 1TB????) 
timeout=15 # default: 10. increase if "The read operation timed out"
#encode_quality=95

img2dataset --url_list $input_folder --input_format "parquet" \
        --url_col "URL" --caption_col "TEXT" \
		--output_format $output_format \
        --output_folder  $output_folder \
		--processes_count 16 --thread_count 64 --image_size 512 \
        --resize_only_if_bigger=True \
		--resize_mode="keep_ratio" \
		--skip_reencode=True \
        --timeout $timeout \
        --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]' \
		#--enable_wandb True

