output_path=output/train_controlnet
rm -rf $output_path
mkdir -p ${output_path:?}
nohup python -u train_controlnet.py  \
  --train_config configs/train/train_config_controlnet_v1.yaml  \
  --data_path dataset/fill50k  \
  --output_path $output_path  \
  --train_batch_size 4 \
  --epochs 4  \
  --init_loss_scale  65536  \
  > $output_path/log_train 2>&1 &

