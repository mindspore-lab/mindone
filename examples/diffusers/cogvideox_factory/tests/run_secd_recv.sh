NUM_NPUS=4
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

msrun --master_port=2234 --worker_num=$NUM_NPUS --local_worker_num=$NUM_NPUS --bind_core=True --log_dir="./log_test_send_recv" --join True ${SCRIPT_DIR}/test_send_recv.py
