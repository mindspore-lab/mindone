# !/usr/bin/bash
# ------------------------------------
#      stage     | Minimum Memory Support
# ------------------------------------
# text_to_image  | ~14G
# ------------------------------------
# image_to_views | lite ~8G
# image_to_views | std ~16G
# ------------------------------------
# views_to_mesh  | ~10G (90000 faces)
# ------------------------------------

# Usage:
# bash scripts/text_to_3d_std_separately.sh 'a lovely rabbit' ./outputs/test
## need Memory > 16G

text_prompt=$1
save_folder=$2

# model name or paths
text2image_path=Tencent-Hunyuan/HunyuanDiT-Diffusers #local: weights/hunyuanDiT hugggingface: Tencent-Hunyuan/HunyuanDiT-Diffusers
std_pretrain=./weights/mvd_std
mv23d_ckt_path=./weights/svrm/svrm.safetensors

# init
use_lite=false
do_texture_mapping=false # not support yet
max_faces_num=90000

mkdir -p $save_folder

python infer/text_to_image.py \
    --text2image_path $text2image_path \
    --text_prompt "$text_prompt" \
    --output_img_path $save_folder/img.jpg \
    --seed 0 \
    --steps 25
&& \
python infer/removebg.py \
    --rgb_path $save_folder/img.jpg \
    --output_rgba_path $save_folder/img_nobg.png \
&& \
python infer/image_to_views.py \
    --mvd_ckt_path $std_pretrain \
    --rgba_path $save_folder/img_nobg.png \
    --output_views_path $save_folder/views.jpg \
    --output_cond_path $save_folder/cond.jpg \
    --seed 0 \
    --steps 50 \
    --use_lite $use_lite \
&& \
python infer/views_to_mesh.py \
    --views_path $save_folder/views.jpg \
    --cond_path $save_folder/cond.jpg \
    --save_folder $save_folder \
    --max_faces_num $max_faces_num \
    --mv23d_cfg_path ./svrm/configs/svrm.yaml \
    --mv23d_ckt_path $mv23d_ckt_path \
    --use_lite $use_lite \
    --do_texture_mapping $do_texture_mapping \
&& \
python infer/gif_render.py \
    --mesh_path $save_folder/mesh.obj \
    --output_gif_path $save_folder/output.gif
