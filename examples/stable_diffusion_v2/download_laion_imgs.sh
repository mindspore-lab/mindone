output_format="files"
output_folder=/data3/datasets/laion_art # make sure this folder is set on the disk with large enough space (> 1TB????) 
#encode_quality=95

img2dataset --url_list /home/yx/datasets/diffusion/laion_art/laion-art.parquet --input_format "parquet" \
        --url_col "URL" --caption_col "TEXT" \
		--output_format $output_format \
        --output_folder  $output_folder \
		--processes_count 16 --thread_count 64 --image_size 512 \
        --resize_only_if_bigger=True \
		--resize_mode="keep_ratio" \
		--skip_reencode=True \
        --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]' \
		#--enable_wandb True

