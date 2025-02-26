# train
python train.py \
    --model_cfg configs/sampling/sv3d_u.yaml \
    --train_cfg configs/sv3d_u_train.yaml \

# eval overfitted ckpt
python simple_video_sample.py \
    --ckpt PATH/TO/YOUR/OVERFITTED/CKPT \
    --input PATH/TO/OVERFITTED/DATA.png \
    --mode 1 \
    --decoding_t 1 \
    --version sv3d_u_overfitted_ckpt
