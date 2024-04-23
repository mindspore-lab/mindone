export MS_DATASET_SINK_QUEUE=10

# enable kbk
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

python train_t2v.py --config configs/train/stdit_512x512x16.yaml \
    --csv_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
    --video_folder datasets/sora_overfitting_dataset_0410 \
    --text_embed_folder datasets/sora_overfitting_dataset_0410 \
    --num_frames=10 \
    --use_recompute=False \
    --enable_flash_attention=True \
    --enable_dvm=True \
    --use_ema=False \
    --batch_size=1 \
    --dataset_sink_mode=False \
    --num_parallel_workers=14 \
    --max_rowsize=96 \
    --output_path outputs/stdit_512x512x16_bs1_c200_faNoPad \

