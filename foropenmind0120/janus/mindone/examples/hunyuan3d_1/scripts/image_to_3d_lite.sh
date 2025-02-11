# image to 3d fast
text2image_path=Tencent-Hunyuan/HunyuanDiT-Diffusers
lite_pretrain=./weights/mvd_lite
mv23d_ckt_path=./weights/svrm/svrm.safetensors
python main.py \
    --mvd_ckt_path $lite_pretrain \
    --mv23d_cfg_path ./svrm/configs/svrm.yaml \
    --mv23d_ckt_path $mv23d_ckt_path \
    --image_prompt ./demos/example_000.png \
    --save_folder ./outputs/test/ \
    --max_faces_num 10000 \
    --use_lite
