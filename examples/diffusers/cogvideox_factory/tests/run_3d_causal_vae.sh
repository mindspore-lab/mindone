#export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Num of NPUs for test
export MS_DEV_RUNTIME_CONF="memory_statistics:True"
# export MS_DEV_LAZY_FUSION_FLAGS="--opt_level=1"
NUM_NPUS=4
# export DEVICE_ID=3

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Prepare launch cmd according to NUM_NPUS
if [ "$NUM_NPUS" -eq 1 ]; then
    cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
    avg=`expr $cpus \/ 8`
    gap=`expr $avg \- 1`
    start=`expr $DEVICE_ID \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    LAUNCHER="taskset -c $cmdopt python"
    EXTRA_ARGS=""
else
    LAUNCHER="msrun --bind_core=True --worker_num=$NUM_NPUS --local_worker_num=$NUM_NPUS --log_dir=log_test_vae --join True"
fi

echo "Start Running:"
cmd="$LAUNCHER ${SCRIPT_DIR}/test_3d_causal_vae.py"
echo "Running command: $cmd"
eval $cmd
echo "=============================================="
