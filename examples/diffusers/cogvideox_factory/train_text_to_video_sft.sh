# export MS_DEV_RUNTIME_CONF="memory_statistics:True,compile_statistics:True"
# Num of NPUs for training
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_NPUS=8

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-5")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("20000")
SP=False
SP_SIZE=$NUM_NPUS
F=77  # Need to change to multiple of 8, when LATENTS_CACHE=0 & SP=True
FA_RCP=False
ENABLE_DYNAMIC_SHAPE=0
LATENTS_CACHE=1
EMBEDDINGS_CACHE=1
OUTPUT_ROOT_DIR=./output_sft

# MindSpore settings
MINDSPORE_MODE=0
JIT_LEVEL=O1
AMP_LEVEL=O2
DEEPSPEED_ZERO_STAGE=3

# Prepare launch cmd according to NUM_NPUS
if [ "$NUM_NPUS" -eq 1 ]; then
    LAUNCHER="python"
    EXTRA_ARGS=""
    SP=False
else
    LAUNCHER="msrun --bind_core=True --worker_num=$NUM_NPUS --local_worker_num=$NUM_NPUS --log_dir="./log_sp_graph""
    EXTRA_ARGS="--distributed --zero_stage $DEEPSPEED_ZERO_STAGE"
fi
if [ "$ENABLE_DYNAMIC_SHAPE" -eq 1 ]; then
  # Enable kernel backoff to support the Python floor operation at line 444 in mindone/mindone/diffusers/models/embeddings.py.
  # Otherwise, a "RuntimeError" will be raised: "The current operator needs to be supplemented with an adapter, please
  # check in `transform` directory. node is Default/network-TrainStepForCogVideo/transformer-CogVideoTransformer3DModel_SP/patch_embed-CogVideoXPatchEmbed/ScalarFloorDiv-op1".
  # Additionally, it is not feasible to replace the Python floor operation with `ms.mint.floor`. The reason is that
  # `ms.mint.floor` does not accept scalar input, the scalar input must be converted to an `ms.Tensor` first. However,
  # `ms.Tensor` does not support non-constant input in graph mode.
  export MS_DISABLE_KERNEL_BACKOFF=0
  EXTRA_ARGS="$EXTRA_ARGS --dynamic_shape --bucket_config=cogvideox/bucket.yaml"
fi
if [ "$LATENTS_CACHE" -eq 1 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --latents_cache"
fi
if [ "$EMBEDDINGS_CACHE" -eq 1 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --embeddings_cache"
fi

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --local-dir /path/to/my/datasets/tom-and-jerry-dataset
DATA_ROOT="preprocessed-dataset"
CAPTION_COLUMN="prompts.txt"
VIDEO_COLUMN="videos.txt"
MODEL_PATH="THUDM/CogVideoX1.5-5b"

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="${OUTPUT_ROOT_DIR}/cogvideox-sft__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="$LAUNCHER cogvideox/cogvideox_text_to_video_sft.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --height_buckets 768 \
          --width_buckets 1360 \
          --frame_buckets $F \
          --max_num_frames $F \
          --gradient_accumulation_steps 1 \
          --dataloader_num_workers 2 \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 1 \
          --seed 42 \
          --mixed_precision bf16 \
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
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --report_to tensorboard \
          --mindspore_mode $MINDSPORE_MODE \
          --jit_level $JIT_LEVEL \
          --amp_level $AMP_LEVEL \
          --enable_sequence_parallelism $SP \
          --sequence_parallel_shards $SP_SIZE \
          $EXTRA_ARGS"

        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
