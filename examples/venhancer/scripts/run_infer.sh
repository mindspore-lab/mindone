python inference.py \
  --input_path "prompts/" \
  --prompt_path "prompts/text_prompts.txt" \
  --model_path "models/venhancer_paper.ckpt" \
  --noise_aug=250 \
  --up_scale=4 \
  --target_fps=24
