# Test_module

This document provides a tutorial on how to run `xxx_forward.py` and `xxx_loss.py`. （Forward and Loss Testing）

## Preparation

Before running the test script, please ensure that you have export the environment variable `PYTHONPATH`
```shell
export PYTHONPATH=/path/to/mindone/example/stable_diffusion_xl/
```
## Run

Please run the following command to test the module

For example, to test ResBlock module forward output

```shell
python  tests/test_modules/ResBlock_forward.py
```
You can get forward output from the console or out.npy.

To test ResBlock module loss output
```shell
python  tests/test_modules/ResBlock_loss.py
```
You can get loss from the console.

You can replace ResBlock with UNetModel, BasicTransformerBlock, VAE-Encoder, etc.
