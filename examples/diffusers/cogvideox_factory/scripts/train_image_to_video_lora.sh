#!/bin/bash
# Package path
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# export MS_DEV_RUNTIME_CONF="memory_statistics:True,compile_statistics:True"
# Num of NPUs for training
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_NPUS=8

# Training Configurations
# Experiment with as many hyperparameters as you want!
MIXED_PRECISION="bf16"
LEARNING_RATES=("2e-5")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw_bf16")
MAX_TRAIN_STEPS=("3000")

FA_RCP=False
OUTPUT_ROOT_DIR=./output_lora

# MindSpore settings
MINDSPORE_MODE=0
JIT_LEVEL=O1
AMP_LEVEL=O2

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --local-dir /path/to/my/datasets/tom-and-jerry-dataset
DATA_ROOT="preprocessed-dataset"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"
MODEL_PATH="THUDM/CogVideoX1.5-5B-I2V"
H=768
W=1360
F=77
MAX_SEQUENCE_LENGTH=224

VAE_CACHE=1
EMBEDDINGS_CACHE=1

# Prepare launch cmd according to NUM_NPUS
if [ "$NUM_NPUS" -eq 1 ]; then
    LAUNCHER="python"
    EXTRA_ARGS=""
    SP=False
else
    LAUNCHER="msrun --bind_core=True --worker_num=$NUM_NPUS --local_worker_num=$NUM_NPUS --log_dir="./log_lora" --join=True"
    EXTRA_ARGS="--distributed"
fi

if [ "$VAE_CACHE" -eq 1 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --vae_cache"
fi
if [ "$EMBEDDINGS_CACHE" -eq 1 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --embeddings_cache"
fi

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="./cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="$LAUNCHER ${SCRIPT_DIR}/cogvideox_image_to_video_lora.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --height_buckets 768 \
          --width_buckets 1360 \
          --frame_buckets 77 \
          --max_num_frames 77 \
          --gradient_accumulation_steps 1 \
          --dataloader_num_workers 2 \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision $MIXED_PRECISION \
          --output_dir $output_dir \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps 2000 \
          --gradient_checkpointing \
          --fa_gradient_checkpointing=$FA_RCP \
          --scale_lr \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 800 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --noised_image_dropout 0.05 \
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
