DEVICE_ID=0 python scripts/v2.0/infer_vae_v2.py \
--ae=configs/opensora-v2-0/ae/hunyuan_vae.yaml \
--dataset.csv_path=PATH_TO_CSV \
--dataset.video_folder=PATH_TO_VIDEO_FOLDER \
--dataset.sample_n_frames=129 \
--dataset.target_size=[256,256] \
--dataset.deterministic_sample=True \
--output_dir=OUTPUT_DIR
