time=$(date +"%Y%m%d%H%M")

export MS_MEMORY_STATISTIC=2
export MS_SUBMODULE_LOG_v="{PYNATIVE:0}"
export MS_ENABLE_NUMA=1
export GLOG_v=2
export GLOG_log_dir=./logs_pynative__$time
export GLOG_logtostderr=0
export DEVICE_ID=1

nohup pytho train.py -c configs/training/sky_video.yaml --mode 1 > pynative$time.log 2>&1 &