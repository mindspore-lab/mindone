#!/bin/sh
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "$SCRIPT_DIR"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
echo "$PROJECT_DIR"
EXAMPLE_DIR="$(dirname "${PROJECT_DIR}")"
echo "$EXAMPLE_DIR"
PACKAGE_DIR="$(dirname "${EXAMPLE_DIR}")"
echo "$PACKAGE_DIR"

export PYTHONPATH="${PROJECT_DIR}:${PACKAGE_DIR}:${PYTHONPATH}"

echo "******** Graph Mode ********"
msrun --master_port=1234 --worker_num=2 --local_worker_num=2 --log_dir="./log_test_sp_graph" --join True "${SCRIPT_DIR}"/test_mmdit_sequence_parallel.py --mode 0
echo "Done. Check the log at './log_test_sp_graph'."
echo "========================================================================="

echo "******** Pynative Mode ********"
msrun --master_port=1235 --worker_num=2 --local_worker_num=2 --log_dir="./log_test_sp_pynative" --join True "${SCRIPT_DIR}"/test_mmdit_sequence_parallel.py --mode 1
echo "Done. Check the log at './log_test_sp_pynative'."
