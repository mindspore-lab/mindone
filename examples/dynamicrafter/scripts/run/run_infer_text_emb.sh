resolution=$1
if [ "${resolution}" != "512" ] && [ "${resolution}" != "1024" ]; then
    echo "ERROR! Input value of 'resolution' is invalid."
    exit 1
fi

# csv_path=prompts/${resolution}/test_prompts.csv
# output_path=prompts/text_emb/${resolution}
csv_path=/root/lhy/data/data/mixkit-100videos/mixkit/video_caption_train.csv
output_path=text_emb/mixkit100
mkdir -p output_path

python scripts/infer_text_emb.py \
--config configs/training_${resolution}_v1.0.yaml \
--csv_path $csv_path \
--output_path $output_path \