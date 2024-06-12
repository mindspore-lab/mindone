time=$(date +"%Y%m%d%H%M")

export MS_MEMORY_STATISTIC=2
export MS_SUBMODULE_LOG_v="{PYNATIVE:0}"
export MS_ENABLE_NUMA=1
export GLOG_v=2
export GLOG_log_dir=./logs_pynative__$time
export GLOG_logtostderr=0
export ASCEND_RT_VISIBLE_DEVICES=6,7

msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=19712 --log_dir=$time \
python train.py -c configs/training/sky_video.yaml --mode 1 --use_parallel True --use_adamzero2 True