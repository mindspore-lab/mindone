# Example usage to train in 1 card in PYNATIVE_MODE.
# To save RAM, set num_parallel_workers=1
export MS_INDEPENDENT_DATASET=True
export MS_ALLOC_CONF="enable_vmm:True;"
export TRAIN_CONFIG="./configs/train-sample-base.yaml"
export DEVICE_ID=0
EPOCH=100000
OUTPUT_PATH=./outputs/train_3sample_1card_e$EPOCH
python -m openlrm.launch train.lrm --config $TRAIN_CONFIG --mode 1 --num_parallel_workers 1 --epochs $EPOCH --loss_scaler_type static --init_loss_scale 16 &> train_${DEVICE_ID}_3sample_1card_e${EPOCH}.log &
echo "Log at train_${DEVICE_ID}_3sample_1card_e${EPOCH}.log"
