ckpt_path=$1
python generation_inference.py \
    --temperature 0.0 \
    --prompt "Jade painting style, on a wooden table, placed two green fox statues, their background is green woods." \
    --ckpt_path $ckpt_path \
