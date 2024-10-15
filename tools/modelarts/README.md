# How to launch distributed training on ModelArts

## Launch with `msrun`

MindSpore >= 2.3 supports using [`msrun`](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html) to launch distributed training on ModelArts.

Usage:
```shell
# $MA_JOB_DIR is the working dir of your training job on modelarts.
export output_path=[YOUR_OUTPUT_PATH]  # should be an ABSOLUTE path
python $MA_JOB_DIR/mindone/tools/modelarts/msrun/msrun.py [YOUR_WORK_DIR] [YOUR_SCRIPT_NAME]
```

Example:
```shell
export output_path=$MA_JOB_DIR/mindone/examples/opensora_hpcai/output
python $MA_JOB_DIR/mindone/tools/modelarts/msrun/msrun.py mindone/examples/opensora_hpcai/scripts train.py
```

⚠️ Note:
- `$output_path` should be an **ABSOLUTE** path.
- If no `$output_path`, the msrun logs will be saved at the ModelArts default output directory: `/home/ma-user/modelarts/outputs/output_path_0/`.
- If you want to update MindSpore version before the training starts, please `export ms_whl=[PATH_TO_MINDSPORE_WHL_FILE]`, and check the log to confirm whether the MindSpore is updated successfully.

## Launch with rank table
Refer to [here](https://support.huaweicloud.com/bestpractice-modelarts/develop-modelarts-0120.html).

Usage example:
```shell
python $MA_JOB_DIR/mindone/tools/modelarts/run_ascend/run_ascend.py python PATH/TO/train.py --config PATH/TO/your_config.yaml
# $MA_JOB_DIR is the working dir of your training job on modelarts.
```
