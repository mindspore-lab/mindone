#!/bin/sh

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
EXAMPLE_DIR="$(dirname "${PROJECT_DIR}")"
PACKAGE_DIR="$(dirname "${EXAMPLE_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PACKAGE_DIR}:${PYTHONPATH}"

echo "Graph Mode:"
msrun --master_port=1234 --worker_num=2 --local_worker_num=2 --log_dir="./log_test_dsp_graph" --join True ${SCRIPT_DIR}/test_stdit3_sequence_parallelism_dsp.py --mode 0
echo "Done. Check the log at './log_test_dsp_graph'."
echo "=============================================="

# echo "Pynative Mode:"
# msrun --master_port=1235 --worker_num=2 --local_worker_num=2 --log_dir="./log_test_sp_pynative" --join True ${SCRIPT_DIR}/test_stdit3_sequence_parallelism.py --mode 1
# echo "Done. Check the log at './log_test_sp_pynative'."
