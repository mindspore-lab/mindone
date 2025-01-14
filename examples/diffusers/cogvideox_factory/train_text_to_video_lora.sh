export MS_ENABLE_NUMA=1

# Num of NPUs for training
NUM_NPUS=8

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-4" "1e-3")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw" "adam")
MAX_TRAIN_STEPS=("3000")

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
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="/path/to/my/datasets/disney-dataset"

CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
MODEL_PATH="THUDM/CogVideoX-5b"

# Set ` --load_tensors ` to load tensors from disk instead of recomputing the encoder process.
# Launch experiments with different hyperparameters

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="./cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="$LAUNCHER training/cogvideox_text_to_video_lora.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --id_token BW_STYLE \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 2 \
          --validation_prompt \"BW_STYLE A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::BW_STYLE A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 10 \
          --seed 42 \
          --lora_rank 128 \
          --lora_alpha 128 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps 1000 \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 400 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --load_tensors \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --report_to tensorboard \
          --mindspore_mode $MINDSPORE_MODE \
          --amp_level $AMP_LEVEL \
          $EXTRA_ARGS"

        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
