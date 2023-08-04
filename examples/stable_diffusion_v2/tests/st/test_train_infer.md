# Test_train_infer

This document provides a tutorial on how to run `test_train_infer_dummy.py`.

## Preparation

Before running the `test_train_infer_dummy.py`, please ensure that the pretrained weight `sd_v2_base-57526ee4.ckpt`
is located in the `path_to_examples/stable_diffusion_v2/models` folder, indicating that
`path_to_examples/stable_diffusion_v2/models/sd_v2_base-57526ee4.ckpt` exists.

## Run

Please run the following command to test `train_text_to_image.py`, `text_to_image.py` and `train_dreambooth.py`.
```shell
pytest tests/st/test_train_infer_dummy.py
```
