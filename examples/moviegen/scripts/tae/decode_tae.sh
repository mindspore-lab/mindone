export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LATENTS_PATH=$1
OUT_DIR=$2
NUM_FRAMES=$3

# Set save format based on number of frames
if [ "$NUM_FRAMES" -eq 1 ]; then
    SAVE_FORMAT="png"
else
    SAVE_FORMAT="mp4"
fi

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$OUT_DIR" \
python scripts/inference_tae.py \
--env.mode 0 \
--env.jit_level O1 \
--env.distributed True \
--tae.pretrained models/tae_ucf101pt_mixkitft-b3b2e364.ckpt \
--tae.use_tile True \
--tae.dtype bf16 \
--num_frames "$NUM_FRAMES" \
--latent_data.folder "$LATENTS_PATH" \
--output_path "$OUT_DIR" \
--save_format "$SAVE_FORMAT"
