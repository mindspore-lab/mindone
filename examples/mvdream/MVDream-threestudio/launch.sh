# train low res
python launch.py \
        --train \

# train high res
python launch.py \
        --train \
        --train_highres \
        resume="CKPT_OUTPUT_ABOVE" \
        system.use_recompute=true \
