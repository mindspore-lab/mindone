#!/bin/sh
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname $(dirname "${SCRIPT_DIR}"))"
EXAMPLE_DIR="$(dirname "${PROJECT_DIR}")"
PACKAGE_DIR="$(dirname "${EXAMPLE_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PACKAGE_DIR}:${PYTHONPATH}"


msrun --master_port=1234 --worker_num=2 --local_worker_num=2 --log_dir="./log_test_sp_data" --join True \
    ${SCRIPT_DIR}/test_sequence_parallel_data.py \
    --config configs/finetune/mixkit_256x256x29.yaml \
    --train.sequence_parallel.shards 2 \

echo "Done. Check the log at './log_test_sp_data'."
