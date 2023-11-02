output_path=output/train_controlnet_10samples_1101
rm -rf $output_path
mkdir -p ${output_path:?}
nohup python -u train_controlnet.py  \
  --train_config configs/train/train_config_controlnet_v1.yaml  \
  --data_path dataset/fill50k_10sample  \
  --train_batch_size 1  \
  --epochs 800  \
  --start_learning_rate 1e-5 \
  --ckpt_save_interval 100  \
  --output_path $output_path  \
  --init_loss_scale  65536  \
  > $output_path/log_train 2>&1 &

