# for ms2.2 0907, remove this later
export MS_ENABLE_REF_MODE=1

# Exp01, different conditions to videos
python infer.py \
    --cfg configs/exp01_vidcomposer_full.yaml \
    --seed 9999 \
    --input_video "demo_video/blackswan.mp4" \
    --input_text_desc "A black swan swam in the water"


# Exp02, motion transfer to videos
python infer.py \
    --cfg configs/exp02_motion_transfer.yaml \
    --seed 9999 \
    --input_video "demo_video/motion_transfer.mp4" \
    --image_path "demo_video/sunflower.png" \
    --input_text_desc "A sunflower in a field of flowers"


python infer.py \
    --cfg configs/exp02_motion_transfer_vs_style.yaml \
    --seed 9999 \
    --input_video "demo_video/motion_transfer.mp4" \
    --image_path "demo_video/moon_on_water.jpg" \
    --style_image "demo_video/moon_on_water.jpg" \
    --input_text_desc "A beautiful big silver moon on the water"


python infer.py \
    --cfg configs/exp02_motion_transfer_vs_style.yaml \
    --seed 888 \
    --input_video "demo_video/motion_transfer.mp4" \
    --image_path "demo_video/style/fangao_01.jpeg" \
    --style_image "demo_video/style/fangao_01.jpeg" \
    --input_text_desc "Beneath Van Gogh's Starry Sky"


# Exp03, single sketch to videos with style
python infer.py \
    --cfg configs/exp03_sketch2video_style.yaml \
    --seed 8888 \
    --sketch_path "demo_video/src_single_sketch.png" \
    --style_image "demo_video/style/qibaishi_01.png" \
    --input_text_desc "Red-backed Shrike lanius collurio"

# Exp04, single sketch to videos without style
python infer.py \
    --cfg configs/exp04_sketch2video_wo_style.yaml \
    --seed 144 \
    --sketch_path "demo_video/src_single_sketch.png" \
    --input_text_desc "A little bird is standing on a branch"


# Exp05, image depth to videos without style
python infer.py \
    --cfg configs/exp05_text_depths_wo_style.yaml \
    --seed 9999 \
    --input_video demo_video/tennis.mp4 \
    --input_text_desc "Ironman is fighting against the enemy, big fire in the background, photorealistic"


# Exp06, image depth to videos with style
python infer.py \
    --cfg configs/exp06_text_depths_vs_style.yaml \
    --seed 9999 \
    --input_video demo_video/tennis.mp4 \
    --style_image "demo_video/style/fangao_01.jpeg" \
    --input_text_desc "Van Gogh played tennis under the stars"
