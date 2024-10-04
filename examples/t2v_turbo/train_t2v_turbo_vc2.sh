# FPS: 16; 512x320

export DEVICE_ID=6
CSV_PATH="csv-path.csv"
DATA_PATH="data-dir/"
OUTPUT_DIR="output-dir/"

python train_t2v_turbo_vc2.py \
  --pretrained_model_path checkpoints/t2v-vc2/t2v_VC2.ckpt \
  --video_rm_ckpt_dir checkpoints/InternVideo2-stage2_1b-224p-f4.ckpt \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --reward_fn_name hpsv2 \
  --reward_scale 0.0 \
  --video_rm_name vi_clip2 \
  --video_reward_scale 0.0 \
  --no_scale_pred_x0 \
  --csv_path $CSV_PATH \
  --data_path $DATA_PATH \
  --max_train_samples 100 \
  --mode 1 \
  --cast_teacher_unet \
  --lora_rank 64 \
  --jit_level O0 \
  --mixed_precision fp16 \
  --n_frames 8 \
  --debug True \
  --reward_batch_size 3 \
  --video_rm_batch_size 4 \
  --learning_rate 2.0e-6 \
  --lr_warmup_steps 100 \
  --loss_type huber \
  --output_dir $OUTPUT_DIR \