
# train low res
python launch.py \
        --train \

# train high res
python launch.py \
        --train \
        --resume_ckpt CKPT_OUTPUT_ABOVE \
        --train_highres \
        system.use_recompute=true \
