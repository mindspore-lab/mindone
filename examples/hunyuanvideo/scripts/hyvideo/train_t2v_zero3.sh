export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=8000 --log_dir="./parallel_logs" \
  scripts/train.py \
   --config configs/train/stage1_t2v_256px.yaml \
   --env.mode 0 \
   --env.distributed True \
   --model.name "HYVideo-T/2-cfgdistill" \
   --train.settings.zero_stage 3 \
   --dataset.csv_path datasets/mixkit-100videos/video_caption_train.csv \
   --dataset.video_folder datasets/mixkit-100videos/mixkit \
   --dataset.text_emb_folder datasets/mixkit-100videos/text_embed \
   --dataset.empty_text_emb datasets/mixkit-100videos/empty_string_text_embeddings.npz \
   --valid.dataset.csv_path datasets/mixkit-100videos/video_caption_test.csv \
   --valid.dataset.video_folder datasets/mixkit-100videos/mixkit \
   --valid.dataset.text_emb_folder datasets/mixkit-100videos/text_embed \
