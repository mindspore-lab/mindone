# train low res
python launch.py \
        --train \

# train high res
python launch.py \
        --train \
        --train_highres \
        resume="PATH/OUTPUTS/CKPT" system.use_recompute=true
