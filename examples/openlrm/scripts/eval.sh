# script to evaluate trained checkpoint
# output inference time, PSNR and SSIM at output_path/epoch/log_0.txt
# evaluation set: in model_path/cfg.yaml ->  dataset.subsets[0].meta_path.val
DEVICE_ID=0 python eval.py --model_path ./outputs/base/2025-01-06T14-37-07 --ckpt_name openlrm-e50000.ckpt --output_path output/base
