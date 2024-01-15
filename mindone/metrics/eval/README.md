
This folder contains code and scripts for diffusion model evaluation, e.g.,

- Fréchet inception distance (FID)
- CLIP score
- CLIP directional similarity
- IS (Inception Score)


Note that all the above metrics are computed based on neural network models.

> A convincing evaluation for diffusion models requires both visually qualitative comparision and quantitative measure. A lower FID score (or a higher CLIP score) does not necessarily show one model is better than another.

## Usage

### FID

To compute the FID between the real images and the generated images, in `examples/stable_diffusion` directory, please run

```
python tools/eval/eval_fid.py --real_dir {dir_to_real_images}  --gen_dir {dir_to_generated_images}
```

By default, we use MindSpore backend for FID computing (to run inception v3 model inference and extract image features). You may swich to `torchmetrics` backend  by setting `--backend=pt`. The computational difference between these two backends is usually lower than 0.1%, which is neglectable.

For more usage, please run `python tools/eval/eval_fid.py -h`.

> In the first time running, it will automatically download the checkpoint file for Inception V3 FID model. If you fail to downalod due to network problem, please manually download it from [this link](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/fid/inception_v3_fid-9ec6dfe4.ckpt) and put it under `~/mindspore/models/`

### CLIP Score

To compute the CLIP score between images and texts, in `examples/stable_diffusion_v2` directory, please run

- Mindspore backend
```
python tools/eval/eval_clip_score.py --ckpt_path <path-to-model> --image_path_or_dir <path-to-image> --prompt_or_path <string/path-to-txt>
```
- PyTorch backend
```
python tools/eval/eval_clip_score.py --backend pt --model_name <HF-model-name> --image_path_or_dir <path-to-image> --prompt_or_path <string/path-to-txt>
```
By default, we use MindSpore backend for CLIP score computing (to run CLIP model inference and extract image & text features). You may swich to use `torchmetrics` by setting `--backend=pt`. The computational difference between these two backends is usually lower than 0.1%, which is neglectable.

For more usage, please run `python tools/eval/eval_clip_score.py -h`.

You need to download the checkpoint file for a CLIP model of your choice. Download links for some models are provided below.

- [clip_vit_b_16](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_b_16.ckpt) (Default)
- [clip_vit_b_32](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/clip_vit_b_32.ckpt)
- [clip_vit_l_14](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_l_14.ckpt)

For other compatible models, e.g., OpenCLIP, you can download `pytorch_model.bin` from HuggingFace (HF) and then convert to `.ckpt` using `tools/_common/clip/convert_weight.py`. When using a model other than the default, you should supply the path to your model's config file. Some useful examples are provided in `tools/_common/clip/configs`.

`image_path` should lead to an image file or a directory containing images. If it is a directory, then the images are sorted by their filename in an ascending order. `prompt` can be either a piece of text or the path to an `.txt` file, where prompts are placed line by line. Images and prompts are matched such that each prompt corresponding to one or many images in order.



## Reference

[1] https://huggingface.co/docs/diffusers/conceptual/evaluation
[2] https://arxiv.org/abs/1606.03498
