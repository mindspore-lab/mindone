# Test_module

This document provides a tutorial on how to run `test_moudule.py`.

## Preparation

Before running the `test_moudule.py`, please ensure that you have export the environment variable `PYTHONPATH`
```shell
export PYTHONPATH=/path/to/mindone/example/stable_diffusion_xl/
```
## Run

Please run the following command to test the module you want to test by the parameter `--net`


For example, to test ResBlock module

```shell
python  tests/st/test_module.py --net ResBlock
```
You can replace ResBlock with UNetModel, BasicTransformerBlock, VAE-Encoder, etc.
