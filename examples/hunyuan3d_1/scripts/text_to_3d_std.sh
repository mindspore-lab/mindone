# text to 3d standard
text2image_path=Tencent-Hunyuan/HunyuanDiT-Diffusers
std_pretrain=./weights/mvd_std
mv23d_ckt_path=./weights/svrm/svrm.safetensors
python main.py \
    --text2image_path "${text2image_path}" \
    --mvd_ckt_path $std_pretrain \
    --mv23d_cfg_path ./svrm/configs/svrm.yaml \
    --mv23d_ckt_path $mv23d_ckt_path \
    --text_prompt "一个广式茶杯" \
    --save_folder ./outputs/test/ \
    --max_faces_num 90000
