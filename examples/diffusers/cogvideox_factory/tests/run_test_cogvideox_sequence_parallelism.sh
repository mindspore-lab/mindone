#!/bin/sh
export MS_DEV_RUNTIME_CONF="memory_statistics:True"
#export MS_DEV_RUNTIME_CONF="memory_statistics:True,compile_statistics:True"
#export ASCEND_RT_VISIBLE_DEVICES=2,3
NUM_NPUS=2
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

echo "Start Running:"
msrun --master_port=1234 --worker_num=$NUM_NPUS --local_worker_num=$NUM_NPUS --bind_core=True --log_dir="./log_test_sp_graph" --join True ${SCRIPT_DIR}/test_cogvideox_sequence_parallelism.py
echo "Done. Check the log at './log_test_sp_graph'."
echo "=============================================="
