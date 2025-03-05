# export MS_DEV_RUNTIME_CONF="memory_statistics:True,compile_statistics:True"
# Num of NPUs for inference
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_NPUS=8
SP=True
SP_SIZE=$NUM_NPUS
DEEPSPEED_ZERO_STAGE=3

# MindSpore settings
MINDSPORE_MODE=0
JIT_LEVEL=O1

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --local-dir /path/to/my/datasets/tom-and-jerry-dataset
MODEL_PATH="THUDM/CogVideoX1.5-5b"
# TRANSFORMER_PATH and LORA_PATH only choose one to set.
TRANSFORMER_PATH=""
PROMPT="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
H=768
W=1360
F=80
MAX_SEQUENCE_LENGTH=224
OUTPUT_DIR=./output_infer_${H}_${W}_${F}
test -d ${OUTPUT_DIR} || mkdir ${OUTPUT_DIR}

if [ "$NUM_NPUS" -eq 1 ]; then
    LAUNCHER="python"
    EXTRA_ARGS=""
    SP=False
else
    LAUNCHER="msrun --bind_core=True --worker_num=$NUM_NPUS --local_worker_num=$NUM_NPUS --log_dir="./log_sp_graph" --join=True"
    EXTRA_ARGS="--distributed --zero_stage $DEEPSPEED_ZERO_STAGE"
    export TOKENIZERS_PARALLELISM=false
fi

cmd="$LAUNCHER cogvideox/infer.py \
    --pretrained_model_name_or_path $MODEL_PATH \
    --prompt \"${PROMPT}\" \
    --transformer_ckpt_path $TRANSFORMER_PATH \
    --height $H \
    --width $W \
    --frame $F \
    --max_sequence_length=$MAX_SEQUENCE_LENGTH \
    --npy_output_path ${OUTPUT_DIR}/npy \
    --video_output_path ${OUTPUT_DIR}/output.mp4 \
    --seed 42 \
    --mixed_precision bf16 \
    --mindspore_mode $MINDSPORE_MODE \
    --jit_level $JIT_LEVEL \
    --enable_sequence_parallelism $SP \
    --sequence_parallel_shards $SP_SIZE \
    $EXTRA_ARGS"

echo "Running command: $cmd"
eval $cmd
