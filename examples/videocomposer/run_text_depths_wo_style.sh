# Only for MS2.1, uncomment the below line if use MS2.1 on 910B
#export MS_ENABLE_GE=1

# since ms2.2 0907
export MS_ENABLE_REF_MODE=1

# Exp05, from text and depth maps to a video
python infer.py  \
    --cfg configs/exp05_text_depths_wo_style.yaml  \
    --seed 9999  \
    --input_video "datasets/webvid5/3.mp4"  \
    --input_text_desc "Sharp knife to cut delicious smoked fish." \
    --resume_checkpoint $1   \
    --ms_mode 0
