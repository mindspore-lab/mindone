# Convert videocomposer pytorch weights to mindspore weights


The `non_ema_141000_no_watermark.pth` can be downloaded from https://www.modelscope.cn/models/damo/VideoComposer/files.

Then, in the videocomposer directory, you can run the following command to convert the weight of the pytorch
model to the weight of the mindspore model.

```
python tools/pt2ms.py --pt_model_path your_model_path
```
