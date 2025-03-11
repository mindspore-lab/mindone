#!/bin/bash
# Package path
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# For more details on the expected data format, please refer to the README.
NUM_NPUS=8
MODEL_NAME_OR_PATH="THUDM/CogVideoX1.5-5b"
DATA_ROOT="/path/to/my/datasets/video-dataset"  # This needs to be the path to the base directory where your videos are located.
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
OUTPUT_DIR="preprocessed-dataset"
HEIGHT_BUCKETS="768"
WIDTH_BUCKETS="1360"
# Need to change to multiple of 8, when training SP=True
FRAME_BUCKETS="77"
MAX_NUM_FRAMES="77"
MAX_SEQUENCE_LENGTH=224
TARGET_FPS=8
BATCH_SIZE=1
DTYPE=bf16
VAE_CACHE=1
EMBEDDINGS_CACHE=1

if [ "$NUM_NPUS" -eq 1 ]; then
    LAUNCHER="python"
    EXTRA_ARGS=""
    export HCCL_EXEC_TIMEOUT=1800
else
    LAUNCHER="msrun --bind_core=True --worker_num=$NUM_NPUS --local_worker_num=$NUM_NPUS --log_dir="./log_data""
    EXTRA_ARGS="--distributed"
fi
if [ "$VAE_CACHE" -eq 1 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --vae_cache"
fi
if [ "$EMBEDDINGS_CACHE" -eq 1 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --embeddings_cache"
fi

# To create a folder-style dataset structure without pre-encoding videos and captions
# For Image-to-Video finetuning, make sure to pass `--save_image_latents`
CMD="\
  $LAUNCHER ${SCRIPT_DIR}/prepare_dataset.py \
      --pretrained_model_name_or_path $MODEL_NAME_OR_PATH \
      --data_root $DATA_ROOT \
      --caption_column $CAPTION_COLUMN \
      --video_column $VIDEO_COLUMN \
      --output_dir $OUTPUT_DIR \
      --height_buckets $HEIGHT_BUCKETS \
      --width_buckets $WIDTH_BUCKETS \
      --frame_buckets $FRAME_BUCKETS \
      --max_num_frames $MAX_NUM_FRAMES \
      --max_sequence_length $MAX_SEQUENCE_LENGTH \
      --target_fps $TARGET_FPS \
      --batch_size $BATCH_SIZE \
      --dtype $DTYPE \
      $EXTRA_ARGS
"

echo "===== Running \`$CMD\` ====="
eval $CMD
echo -ne "===== Finished running script =====\n"
