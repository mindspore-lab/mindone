# AnimateDiff based on MindSpore

This repository is the MindSpore implementation of [AnimateDiff](https://arxiv.org/abs/2307.04725).

## Features

- [x] Text-to-video generation with AnimdateDiff v2, supporting 16 frames @512x512 resolution on Ascend 910B, 16 frames @256x256 resolution on GPU 3090
- [x] MotionLoRA inference
- [ ] Motion Module Training

## Requirements

```
pip install -r requirements.txt
```

In case `decord` package is not available in your environment, try `pip install eva-decord`


## Prepare Model Weights

First, download the torch pretrained weights referring to [torch animatediff checkpoints](https://github.com/guoyww/AnimateDiff/blob/main/__assets__/docs/animatediff.md#download-base-t2i--motion-module-checkpoints).

- Convert SD dreambooth model
```
cd ../examples/stable_diffusion_v2
python tools/model_conversion/convert_weights.py  --source ../animatediff/models/torch_ckpts/toonyou_beta3.safetensors   --target models/toonyou_beta3.ckpt  --model sdv1  --source_version pt
```

- Convert Motion Module
```
cd ../examples/animatediff/tools
python motion_module_convert.py --src ../torch_ckpts/mm_sd_v15_v2.ckpt --tar ../models/motion_module
```

- Convert Motion LoRA
```
cd ../examples/animatediff/tools
python motion_lora_convert.py --src ../torch_ckpts/mm_sd_v15_v2.ckpt --tar ../models/motion_module
```


## Inference

### Text-to-Video

- Running On Ascend 910\*:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou.yaml --L 16 --H 512 --W 512
```

By default, DDIM sampling is used, and the sampling speed is 1.07s/iter.

Results:



- Running on GPU:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou.yaml --L 16 --H 256 --W 256 --target_device GPU
```

### Motion LoRA
- Running On Ascend 910\*:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou-MotionLoRA.yaml --L 16 --H 512 --W 512
```

By default, DDIM sampling is used, and the sampling speed is 1.07s/iter.
Results:


- Running on GPU:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou-MotionLoRA.yaml --L 16 --H 256 --W 256 --target_device GPU

```
