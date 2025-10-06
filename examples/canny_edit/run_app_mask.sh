export DEVICE_ID=0
export no_proxy="localhost,127.0.0.1"

python app_mask.py \
--sam_checkpoint ./checkpoints/sam2.1_hiera_large.pt \
--model_cfg ./configs/sam2.1/sam2.1_hiera_l.yaml
