# Example usage to train in 1 card in PYNATIVE_MODE.
# To save RAM, set num_parallel_workers=1
export MS_INDEPENDENT_DATASET=True
export MS_ALLOC_CONF="enable_vmm:True;"
export TRAIN_CONFIG="./config/train-sample-large.yaml"
export DEVICE_ID=0
EPOCH=100000
OUTPUT_PATH=./outputs/train_3sample_1card_e$EPOCH
python -m openlrm.launch train.lrm --config $INFER_CONFIG --mode 1 --amp_level O2 --use_recompute True --dtype bf16 --loss_scaler_type --static init_loss_scale 16 --num_parallel_workers 1 --epochs $EPOCH &> train_${DEVICE_ID}_3sample_1card_e${EPOCH}.log &
echo "Log at train_${DEVICE_ID}_3sample_1card_e${EPOCH}.log"
