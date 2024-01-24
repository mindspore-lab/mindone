export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
# For best performance, set to a SD1.5 checkpoint trained from image-finetuning in 256x256 resolution
# pretrained_model_path=models/stable_diffusion/sd_v1.5-d0ab7146.ckpt
# pretrained_model_path=tmp_outputs/sdgen_overfit/2024-01-09T18-40-59/ckpt/sd-1000.ckpt

python train.py --config configs/training/mmv2_train.yaml

# --pretrained_model_path=$pretrained_model_path \
