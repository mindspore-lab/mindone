# How to launch distributed training on ModelArts

## Launch with `msrun`

MindSpore >= 2.3 supports using [`msrun`](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html) to launch distributed training on ModelArts.

Usage example:
```shell
# $MA_JOB_DIR is the working dir of your training job on modelarts.
python $MA_JOB_DIR/mindone/tools/modelarts/msrun/msrun.py [WORK_DIR] [SCRIPT_NAME]
python $MA_JOB_DIR/mindone/tools/modelarts/msrun/msrun.py mindone/examples/opensora_hpcai/scripts train.py
```


## Launch with rank table
Refer to [here](https://support.huaweicloud.com/bestpractice-modelarts/develop-modelarts-0120.html).

Usage example:
```shell
python $MA_JOB_DIR/mindone/tools/modelarts/run_ascend/run_ascend.py python PATH/TO/train.py --config PATH/TO/your_config.yaml
# $MA_JOB_DIR is the working dir of your training job on modelarts.
```
