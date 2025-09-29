export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

output_dir=output/stage1_i2v/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8220 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" \
	python scripts/v2.0/train_v2.py \
	--config=configs/opensora-v2-0/train/stage1_i2v.yaml \
  --env.distributed=True \
  --dataset.csv_path=PATH_TO_CSV \
  --dataset.video_folder=PATH_TO_VIDEO_FOLDER \
  --dataset.text_emb_folder.t5=T5_EMB_FOLDER \
  --dataset.text_emb_folder.clip=CLIP_EMB_FOLDER \
  --dataset.empty_text_emb.t5=EMPTY_T5_EMB \
  --dataset.empty_text_emb.clip=EMPTY_CLIP_EMB \
  --save.output_path="$output_dir"
