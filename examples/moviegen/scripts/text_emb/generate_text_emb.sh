export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CSV_PATH=$1
OUT_PATH=$2

mkdir -p "$OUT_PATH"/byt5_emb
mkdir -p "$OUT_PATH"/ul2_emb

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$OUT_PATH"/byt5_emb --join=True \
python scripts/inference_text_enc.py \
--env.distributed True \
--model_name google/byt5-small \
--prompts_file "$CSV_PATH" \
--output_path "$OUT_PATH"/byt5_emb \
--model_max_length 128

msrun --bind_core=True --master_port=8210 --worker_num=8 --local_worker_num=8 --log_dir="$OUT_PATH"/ul2_emb --join=True \
python scripts/inference_text_enc.py \
--env.distributed True \
--model_name google/ul2 \
--prompts_file "$CSV_PATH" \
--output_path "$OUT_PATH"/ul2_emb \
--model_max_length 512
