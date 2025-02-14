resolution=$1
csv_path=$2
output_path=$3

if [ "${resolution}" != "512" ] && [ "${resolution}" != "1024" ]; then
    echo "ERROR! Input value of 'resolution' is invalid."
    exit 1
fi

mkdir -p $output_path

python scripts/infer_text_emb.py \
--config configs/training_${resolution}_v1.0.yaml \
--csv_path $csv_path \
--output_path $output_path \