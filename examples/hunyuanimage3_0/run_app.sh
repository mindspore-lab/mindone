# ==========================================================================
JOBS_DIR=$(dirname "$0")
PROJECT_BASE=$(cd ${JOBS_DIR} || exit; pwd)
echo "PROJECT_BASE: ${PROJECT_BASE}"
# Startup path
cd ${PROJECT_BASE} || exit 1
export PYTHONPATH=${PROJECT_BASE}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=False
# ==========================================================================
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${WORLD_SIZE:-8}

# Input argument
NPUS=${NPUS:-0,1,2,3,4,5,6,7}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-5000}
MODEL_ID=${MODEL_ID:-"HunyuanImage-3/"}

# App entry point
entry_file="app/run_chatbot.py"

# Clear proxy
export http_proxy=
export https_proxy=
# Avoiding the 'timeout error' in httpx used by gradio. Also, gradio>=4.21.0 is required.
export GRADIO_ANALYTICS_ENABLED=False
export ASCEND_RT_VISIBLE_DEVICES="$NPUS"

# Launch App
msrun --worker_num=${NPROC_PER_NODE} \
    --local_worker_num=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --log_dir="logs/app" \
    --join=True \
    ${entry_file} \
    --open-sidebar \
    --host ${HOST} \
    --port ${PORT} \
    --model-id "${MODEL_ID}" \
    "$@"
