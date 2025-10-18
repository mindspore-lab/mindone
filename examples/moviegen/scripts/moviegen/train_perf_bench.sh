export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# log level
export GLOG_v=2

MODE=0

# ---------------------------- 1B ----------------------------

# Stage 1
output_dir=output/test/stage1_t2i_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage1_t2i_256px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.name llama-1B \
  --model.recompute_every_nth_block "" \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/tae_latents_images \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob 0 \
  --dataset.deterministic_sample=True \
  --dataloader.batch_size=10 \
  --valid.dataset "" \
  --train.ema "" \
  --train.save.ckpt_save_policy=latest_k \
  --train.output_path "$output_dir" \
  --train.steps=500

echo "Completed 1B stage 1: $output_dir"
rm -rf "$output_dir"/ckpt

# Stage 2
output_dir=output/test/stage2_t2iv_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8220 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage2_t2iv_256px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.name llama-1B \
  --model.recompute_every_nth_block "" \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/tae_latents \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob=0 \
  --dataset.deterministic_sample=True \
  --train.ema "" \
  --train.settings.gradient_accumulation_steps=5 \
  --train.output_path "$output_dir" \
  --train.steps=300

echo "Completed 1B stage 2: $output_dir"
rm -rf "$output_dir"/ckpt

# Stage 3
output_dir=output/test/stage3_t2iv_768px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8230 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage3_t2iv_768px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.name llama-1B \
  --model.not_recompute_fa True \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/high_tae_latents \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob=0 \
  --dataset.deterministic_sample=True \
  --train.ema "" \
  --train.settings.gradient_accumulation_steps=5 \
  --train.output_path "$output_dir" \
  --train.steps=30

echo "Completed 1B stage 3: $output_dir"
rm -rf "$output_dir"/ckpt


# ---------------------------- 5B ----------------------------

# Stage 1
output_dir=output/test/stage1_t2i_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8210 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage1_t2i_256px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --train.settings.zero_stage 3 \
  --model.recompute_every_nth_block "" \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/tae_latents_images \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob 0 \
  --dataset.deterministic_sample=True \
  --dataloader.batch_size=10 \
  --valid.dataset "" \
  --train.ema "" \
  --train.save.ckpt_save_policy=latest_k \
  --train.output_path "$output_dir" \
  --train.steps=300

echo "Completed 5B stage 1: $output_dir"
find "$output_dir" -name '*.ckpt' -type f -delete

# Stage 2
output_dir=output/test/stage2_t2iv_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8220 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage2_t2iv_256px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --train.settings.zero_stage 2 \
  --model.not_recompute_fa True \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/tae_latents \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob=0 \
  --dataset.deterministic_sample=True \
  --train.ema "" \
  --train.settings.gradient_accumulation_steps=5 \
  --train.output_path "$output_dir" \
  --train.steps=200

echo "Completed 5B stage 2: $output_dir"
rm -rf "$output_dir"/ckpt

# Stage 3
output_dir=output/test/stage3_t2iv_768px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8230 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage3_t2iv_768px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --train.settings.zero_stage 2 \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/high_tae_latents \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob=0 \
  --dataset.deterministic_sample=True \
  --train.ema "" \
  --train.settings.gradient_accumulation_steps=5 \
  --train.output_path "$output_dir" \
  --train.steps=10

echo "Completed 5B stage 3: $output_dir"
rm -rf "$output_dir"/ckpt


# ---------------------------- 30B ----------------------------

# Stage 1
output_dir=output/test/stage1_t2i_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8210 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage1_t2i_256px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.name=llama-30B \
  --train.settings.zero_stage 3 \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/tae_latents_images \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob 0 \
  --dataset.deterministic_sample=True \
  --dataloader.batch_size=10 \
  --valid.dataset "" \
  --train.optimizer.name adamw_re \
  --train.ema "" \
  --train.save.ckpt_save_policy=latest_k \
  --train.output_path "$output_dir" \
  --train.steps=100

echo "Completed 30B stage 1: $output_dir"
find "$output_dir" -name '*.ckpt' -type f -delete

# Stage 2
output_dir=output/test/stage2_t2iv_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8220 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage2_t2iv_256px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.name=llama-30B \
  --train.settings.zero_stage 3 \
  --train.sequence_parallel.shards 8 \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/tae_latents \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob 0 \
  --dataset.sample_n_frames 32 \
  --dataset.deterministic_sample=True \
  --dataloader.batch_size 1 \
  --train.optimizer.name adamw_re \
  --train.ema "" \
  --train.output_path "$output_dir" \
  --train.steps=100

echo "Completed 30B stage 2: $output_dir"
find "$output_dir" -name '*.ckpt' -type f -delete

# Stage 3
output_dir=output/test/stage3_t2iv_768px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8230 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" --join=True \
python scripts/train.py \
  --config configs/train/stage3_t2iv_768px.yaml \
  --env.mode $MODE \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.name=llama-30B \
  --train.settings.zero_stage 3 \
  --train.sequence_parallel.shards 8 \
  --dataset.csv_path ../../../datasets/mixkit-100videos/video_caption_train_updated.csv \
  --dataset.video_folder ../../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../../datasets/mixkit-100videos/high_tae_latents \
  --dataset.text_emb_folder.ul2 ../../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.text_drop_prob=0 \
  --dataset.sample_n_frames 32 \
  --dataset.deterministic_sample=True \
  --dataloader.batch_size 1 \
  --train.optimizer.name adamw_re \
  --train.ema "" \
  --train.output_path "$output_dir" \
  --train.steps=10

echo "Completed 30B stage 3: $output_dir"
find "$output_dir" -name '*.ckpt' -type f -delete
