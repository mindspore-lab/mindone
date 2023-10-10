export MS_ASCEND_CHECK_OVERFLOW_MODE=1 # for ms+910B, check overflow

# Exp02, Motion Transfer from a video to a Single Image
<<com
python infer.py\
    --cfg configs/exp02_motion_transfer.yaml\
    --seed 9999\
    --input_video "datasets/webvid5/1.mp4"\
    --image_path "datasets/webvid5/vid1_frm1.png"\
    --input_text_desc "Disco light leaks disco ball light reflections shaped rectangular and line with motion blur effect." \
    --resume_checkpoint $1 \
    --ms_mode 0
com

python infer.py\
    --cfg configs/exp02_motion_transfer.yaml\
    --seed 9999\
    --input_video "datasets/webvid5/2.mp4"\
    --image_path "datasets/webvid5/vid2_frm1.png"\
    --input_text_desc "Cloudy moscow kremlin time lapse" \
    --resume_checkpoint $1 \
    --ms_mode 0

