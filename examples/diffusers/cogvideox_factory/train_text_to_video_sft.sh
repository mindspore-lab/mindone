export MS_ENABLE_NUMA=1

# Num of NPUs for training
NUM_NPUS=8

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-4")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("20000")

OUTPUT_ROOT_DIR=/path/to/save/output

# MindSpore settings
MINDSPORE_MODE=1
JIT_LEVEL=O0
AMP_LEVEL=O2
DEEPSPEED_ZERO_STAGE=2

# Prepare launch cmd according to NUM_NPUS
if [ "$NUM_NPUS" -eq 1 ]; then
    LAUNCHER="python"
    EXTRA_ARGS=""
else
    LAUNCHER="msrun --worker_num=$NUM_NPUS --local_worker_num=$NUM_NPUS"
    EXTRA_ARGS="--distributed --zero_stage $DEEPSPEED_ZERO_STAGE"
fi

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --local-dir /path/to/my/datasets/tom-and-jerry-dataset
DATA_ROOT="/path/to/my/datasets/tom-and-jerry-dataset"
CAPTION_COLUMN="captions.txt"
VIDEO_COLUMN="videos.txt"

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="${OUTPUT_ROOT_DIR}/cogvideox-sft__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="$LAUNCHER training/cogvideox_text_to_video_sft.py \
          --pretrained_model_name_or_path THUDM/CogVideoX-5b \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 2 \
          --validation_prompt \"Tom, the mischievous gray cat, is sprawled out on a vibrant red pillow, his body relaxed and his eyes half-closed, as if he's just woken up or is about to doze off. His white paws are stretched out in front of him, and his tail is casually draped over the edge of the pillow. The setting appears to be a cozy corner of a room, with a warm yellow wall in the background and a hint of a wooden floor. The scene captures a rare moment of tranquility for Tom, contrasting with his usual energetic and playful demeanor:::A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps 2000 \
          --gradient_accumulation_steps 4 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 800 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --report_to tensorboard \
          --mindspore_mode $MINDSPORE_MODE \
          --jit_level $JIT_LEVEL \
          --amp_level $AMP_LEVEL \
          $EXTRA_ARGS"

        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
