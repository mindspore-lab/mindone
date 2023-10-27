# Only for MS2.1, uncomment the below line if use MS2.1 on 910B
#export MS_ENABLE_GE=1

# since ms2.2 0907
export MS_ENABLE_REF_MODE=1

# Exp02, Motion Transfer from a video to a Single Image
python infer.py\
    --cfg configs/exp02_motion_transfer.yaml\
    --seed 9999\
    --input_video "datasets/webvid5/1.mp4"\
    --image_path "datasets/webvid5/vid1_frm1.png"\
    --input_text_desc "Disco light leaks disco ball light reflections shaped rectangular and line with motion blur effect." \
    --resume_checkpoint $1 \
    --ms_mode 0


<<com
python infer.py\
    --cfg configs/exp02_motion_transfer.yaml\
    --seed 9999\
    --input_video "datasets/webvid5/2.mp4"\
    --image_path "datasets/webvid5/vid2_frm1.png"\
    --input_text_desc "Cloudy moscow kremlin time lapse" \
    --resume_checkpoint $1 \
    --ms_mode 0
