export MS_DATASET_SINK_QUEUE=10

# enable kbk
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

output_dir=outputs/stdit_256x256x16_fa_debug

# mpirun --allow-run-as-root -n 8 --output-filename $output_dir/logs --merge-stderr-to-stdout \
    python train_t2v.py --config configs/train/stdit_256x256x16.yaml \
        --csv_path "../videocomposer/datasets/webvid5/video_caption.csv" \
        --video_folder "../videocomposer/datasets/webvid5" \
        --text_embed_folder "../videocomposer/datasets/webvid5" \
        --enable_flash_attention=True \
        --enable_dvm=True \
        --num_parallel_workers=14 \
        --max_rowsize=96 \
        --output_path=$output_dir  \
        # --use_parallel=True \
        # --csv_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
        # --video_folder datasets/sora_overfitting_dataset_0410 \
        # --text_embed_folder datasets/sora_overfitting_dataset_0410 \
