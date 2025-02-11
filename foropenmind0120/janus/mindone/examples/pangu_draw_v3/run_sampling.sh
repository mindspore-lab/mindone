export MS_PYNATIVE_GE=1
export current_dir=path/to/mindone/examples/pangu_draw_v3
export PYTHONPATH=$current_dir:$PYTHONPATH
cd $current_dir

# run script
# When the device is running low on memory, the '--offload' parameter might be effective.
python pangu_sampling.py \
--device_target "Ascend" \
--ms_mode 1 \
--ms_amp_level "O2" \
--config "configs/inference/pangu_sd_xl_base.yaml" \
--high_solution \
--weight "path/to/low_timestamp_model.ckpt" \
--high_timestamp_weight "path/to/high_timestamp_model.ckpt" \
--prompts_file "prompts.txt"
