resolution=$1 # 1024, 512, 256
ckpt=$2

seed=123
name=dynamicrafter_${resolution}_seed${seed}
prompt_dir=prompts/${resolution}/
config=configs/inference_${resolution}_v1.0.yaml
res_dir="results"

if [ "$resolution" = "256" ]; then
    H=256
    W=256
    FS=3  ## This model adopts frame stride=3, range recommended: 1-6 (larger value -> larger motion)
elif [ "$resolution" = "512" ]; then
    H=320
    W=512
    FS=24 ## This model adopts FPS=24, range recommended: 15-30 (smaller value -> larger motion)
elif [ "$resolution" = "1024" ]; then
    H=576
    W=1024
    FS=10 ## This model adopts FPS=10, range recommended: 15-5 (smaller value -> larger motion)
else
    echo "Invalid resolution input. Please enter 256, 512, or 1024."
    exit 1
fi

if [ "$resolution" = "256" ]; then
python scripts/inference.py \
    --device_target Ascend \
    --seed $seed \
    --bs 1 \
    --height $H \
    --width $W \
    --prompt_dir $prompt_dir \
    --config $config \
    --savedir $res_dir/$name \
    --mode 0 \
    --jit_level O1 \
    --n_samples 1 \
    --unconditional_guidance_scale 7.5 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --text_input \
    --video_length 16 \
    --frame_stride $FS \
    --ckpt_path $ckpt
else
python scripts/inference.py \
    --device_target Ascend \
    --seed $seed \
    --bs 1 \
    --height $H \
    --width $W \
    --prompt_dir $prompt_dir \
    --config $config \
    --savedir $res_dir/$name \
    --mode 0 \
    --jit_level O1 \
    --n_samples 1 \
    --unconditional_guidance_scale 7.5 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --text_input \
    --video_length 16 \
    --frame_stride $FS \
    --ckpt_path $ckpt \
    --timestep_spacing 'uniform_trailing' \
    --guidance_rescale 0.7 \
    --perframe_ae
fi
