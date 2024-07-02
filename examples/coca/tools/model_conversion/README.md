# Model Conversion from Torch to mindspore

We provide scripts of the checkpoint conversion from Torch to MindSpore.

step2. Download the [Official](https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/tree/main), with Mindspore checkpoints.
pre-train weights `open_clip_pytorch_model.bin`  from huggingface.

step2. Convert weight to MindSpore .ckpt format and put it to ./models/.

```shell
python tools/model_conversion/coca_convert.py --source YOUR_TORCH_WEIGHT_PATH --target MINDSPORE-WEIGHT-DIR
```
