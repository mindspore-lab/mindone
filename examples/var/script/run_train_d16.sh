export ASCEND_RT_VISIBLE_DEVICES=6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

export MS_DATASET_SINK_QUEUE=4

# enable kbk: 1
# export MS_ENABLE_ACLNN=0
# export GRAPH_OP_RUN=0

# log level
export GLOG_v=2

output_dir=outputs/d16_train

msrun --bind_core=True --master_port=8200 --worker_num=2 --local_worker_num=2 --log_dir=$output_dir  \
	python train.py --data_path "./data/ImageNet2012" \
    --use_parallel=True \
    --max_device_memory="59GB" \
    --batch_size=96 \
    --tblr=1e-3 \
    --alng=1e-3 \
    --wpe=0.1 \
    --epoch=500 \
    --clip_grad=True \
    --output_path=$output_dir \
