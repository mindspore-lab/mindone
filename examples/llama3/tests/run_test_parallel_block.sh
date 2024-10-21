#!/bin/sh

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
EXAMPLE_DIR="$(dirname "${PROJECT_DIR}")"
PACKAGE_DIR="$(dirname "${EXAMPLE_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PACKAGE_DIR}:${PYTHONPATH}"

LOGDIR="./log_test_parallel_block_graph"
echo "Graph Mode:"
msrun --master_port=1234 --worker_num=2 --local_worker_num=2 --log_dir=$LOGDIR --join True ${SCRIPT_DIR}/test_parallel_block.py --mode 0
echo "Done. Check the log at '$LOGDIR'."
