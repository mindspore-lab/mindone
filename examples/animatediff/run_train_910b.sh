export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

python train.py --config configs/training/mmv2_train.yaml --output_dir outputs/train_infnan_bs4_ds
