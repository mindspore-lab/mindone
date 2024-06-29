seed=123
name=dynamicrafter_512_seed${seed}
res_dir="results"

python scripts/inference.py \
    --device_target Ascend \
    --seed ${seed} \
    --bs 1 \
    --height 320 \
    --width 512 \
    --prompt_dir prompts/512/ \
    --config configs/inference_512_v1.0.yaml \
    --savedir $res_dir/$name \
    --mode 1 \
    --n_samples 1 \
    --unconditional_guidance_scale 7.5 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --text_input \
    --video_length 16 \
    --frame_stride 24 \
    --ckpt_path /home/mindocr/lhy/ckpt/DynamiCrafter/ms_ckpt/model_512.ckpt \
    --timestep_spacing 'uniform_trailing' \
    --guidance_rescale 0.7 \
    --perframe_ae \
