export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"

RATIO=(0.5 0.52 0.57 0.6 0.68 0.72 0.78 0.82 0.88 0.94 1.0 1.07 1.13 1.21 1.29 1.38 1.46 1.67 1.75 1.91 2.0 2.09 2.4 2.5 2.89 3.0 1.0_768 1.0_512)

for item in "${RATIO[@]}"
do
  python demo/sampling_without_streamlit.py \
  --config configs/training/sd_xl_base_finetune_multi_aspect.yaml \
  --save_path path/to/inference \
  --weight path/to/trained/weights \
  --prompt "Build a bright kingdom to stand against the darkness" \
  --sd_xl_base_ratios $item ;
done
