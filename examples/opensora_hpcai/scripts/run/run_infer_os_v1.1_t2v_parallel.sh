export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python scripts/inference_i2v.py --config configs/opensora-v1-1/inference/t2v.yaml \
        --ckpt_path outputs/stdit2_c200_576x1024x24/STDiT-e900.ckpt \
        --prompt_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
        --image_size 576 1024 \
        --num_frames 24 \
        --vae_micro_batch_size 8 \
        --loop 1 \
        --use_parallel=True \

        # --dtype bf16 \

