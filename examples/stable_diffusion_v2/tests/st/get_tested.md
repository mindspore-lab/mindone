# Test Overfit

This document provides a tutorial on how to run `test_overfit.py` for testing `stable diffusion 1.5`.

## Preparation

Before running the `test_vanilla_8p`, please ensure that the `chinese_art_blip` dataset
is located in the `path_to_examples/stable_diffusion_v2/datasets/` folder.
The `chinese_art_blip` dataset can be downloaded [here](https://openi.pcl.ac.cn/attachments/c1941496-fafc-4074-be7b-75fa9f803a53?type=1).


## Supported Platforms & Version

This test example is mainly developed and tested on Ascend 910* platforms with MindSpore framework.
The compatible framework version that are well-tested are listed as follows.

<div align="center">

| Ascend    |  MindSpore   | CANN   | driver |  firmware |
|:-----------:|:----------------:|:--------:|:---------:|:---------:|
| 910*      |     2.2.10 (20231120)    |   7.1.0.1.118  | 23.0.rc3.5.b050   | 7.1.0.1.118|

</div>

## Run

Please run the following command to test `vanilla_8p`, `vanilla`, `lora` or `dreambooth` method.
```shell
pyhon tests/st/test_overfit.py --task {task} --version {version} --device_num {device_num}
```
For more usage, please run `python tests/st/test_overfit.py -h`.
