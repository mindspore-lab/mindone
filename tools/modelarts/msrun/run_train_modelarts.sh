# export HCCL_CONNECT_TIMEOUT=7200
# export HCCL_EXEC_TIMEOUT=7200
#export MS_MEMORY_STATISTIC=1
# export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
export MS_ENABLE_NUMA=1

#export ASCEND_GLOBAL_LOG_LEVEL=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_EVENT_ENABLE=1
master_addr=$1
node_rank=$2
work_dir=$3
script_name=$4
shift
shift
shift
shift

#export GLOG_v=1 MS_SUBMODULE_LOG_v="{RUNTIME_FRAMEWORK:0}"


#export MS_SIMULATION_LEVEL=1


unset RANK_TABLE_FILE
unset RANK_ID

# env

# sed -i "s/\        if get_real_rank() % 8 == 0:/\        if get_real_rank() % 8 == 0 or get_real_rank() % 8 == 7:/g" /home/ma-user/modelarts/user-job-dir/${work_dir}/mindformers/tools/check_rules.py

# if [[ $MS_SIMULATION_LEVEL == 1 ]]; then
# 	export MS_MEMORY_STATISTIC=2
# 	#export MS_MEMORY_TRACE_PATH=/cache/
# 	python /home/ma-user/modelarts/user-job-dir/${work_dir}/run_mindformer.py $@
# else
# 	msrun --worker_num=$RANK_SIZE --local_worker_num=8  --master_addr=$master_addr --node_rank=$node_rank --log_dir=/home/ma-user/modelarts/user-job-dir/device --join=False /home/ma-user/modelarts/user-job-dir/${work_dir}/run_mindformer.py $@
# fi


if [ -z "${ms_whl}" ]; then
    echo "Keep MindSpore version unchanged in the docker container."
else
    echo "Updating MindSpore version from ${ms_whl}..."
    pip uninstall -y mindspore
    pip install ${ms_whl}/mindspore-*.whl
fi


if [ -z "${output_path}" ]; then
    log_root_dir="/home/ma-user/modelarts/outputs/output_path_0"  # no $output_path , set to ModelArts default output directory
else
    log_root_dir="${output_path}"
fi

current=`date "+%Y-%m-%dT%H-%M-%S"`
log_dir=${log_root_dir}/${current}_msrun_log
mkdir -p $log_dir
echo "msrun logs will be saved at: ${log_dir}"

msrun --bind_core=True --worker_num=$(($VC_WORKER_NUM*8)) --local_worker_num=8  --master_addr=$master_addr --node_rank=$node_rank --log_dir=$log_dir --join=False /home/ma-user/modelarts/user-job-dir/${work_dir}/${script_name} $@

sleep 10

# if [ ! node_rank ]; then
# 	#tail -f /home/ma-user/modelarts/user-job-dir/device/worker_0.log | grep -wv -e EVENT -e 'loss: ' -e 'samples/s/p' -e WARNING &
#    tail -f /home/ma-user/modelarts/user-job-dir/device/worker_7.log
# else
# 	tail -f /home/ma-user/modelarts/user-job-dir/device/worker_$((node_rank*8+7)).log
# fi

tail -f $log_dir/worker_$((node_rank*8)).log
