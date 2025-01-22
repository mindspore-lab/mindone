export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

output_dir=PATH_TO_OUT_DIR

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" \
python scripts/inference_tae.py \
--env.mode 0 \
--env.jit_level O1 \
--env.distributed True \
--tae.pretrained models/tae_ucf101pt_mixkitft-b3b2e364.ckpt \
--tae.use_tile True \
--tae.dtype bf16 \
--video_data.folder PATH_TO_IN_DIR \
--video_data.sample_n_frames -1 \
--video_data.size [192,340] \
--output_path "$output_dir"
