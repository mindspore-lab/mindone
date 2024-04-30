# How to launch distributed training on ModelArts

## Launch with `msrun`

MindSpore >= 2.3 supports using [`msrun`](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html) to launch distributed training on ModelArts.

Usage example:

```shell
# $MA_JOB_DIR is the working dir of your training job on modelarts.
python $MA_JOB_DIR/mindone/mindone/utils/modelarts/msrun.py [WORK_DIR] [SCRIPT_NAME]
python $MA_JOB_DIR/mindone/mindone/utils/modelarts/msrun.py mindone/examples/opensora_cai train_t2v.py
```

> If you want to update some package dependencies on modelarts environment right before launching training, please refer to [`ma-pre-start.sh`](./ma-pre-start.sh) and execute it in [`msrun.py`](./msrun.py).

## Launch with rank table
Refer to [here](https://support.huaweicloud.com/bestpractice-modelarts/develop-modelarts-0120.html).

Usage example:
```shell
python $MA_JOB_DIR/mindone/mindone/utils/modelarts/run_ascend.py python PATH/TO/train.py --config PATH/TO/your_config.yaml
# $MA_JOB_DIR is the working dir of your training job on modelarts.
```
