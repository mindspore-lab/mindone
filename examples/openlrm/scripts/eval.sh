# script to evaluate trained checkpoint
# output inference time, PSNR and SSIM at output_path/epoch/log_0.txt
# evaluation set: in model_path/cfg.yaml ->  dataset.subsets[0].meta_path.val

MODEL_PATH=outputs/base/2025-01-06T14-37-07 # <REPLACE_WITH_EXACT_DIR>
CKPT_NAME=openlrm-e50000.ckpt # <REPLACE_WITH_EXACT_NAME>
OUTPUT_PATH=output/base # <REPLACE_WITH_EXACT_OUTPUT_DIR>
DEVICE_ID=0 python eval.py --model_path $MODEL_PATH --ckpt_name $CKPT_NAME --output_path $OUTPUT_PATH
