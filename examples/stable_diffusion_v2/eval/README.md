
This folder contains code and scripts for diffusion model evaluation, including 

- Fréchet inception distance (FID)
- CLIP score
- CLIP directional similarity 


Note that all the above metrics are computed based on neural network models.

> A convincing evaluation for diffusion models requires both visually qaunlitative comparision and quantitative measure. A higher FID score (or CLIP score) does not nessarily show one model is better than another. 

## Usage

### FID

To compute the FID between the real images and the generated images, please run 

```
python eval/eval_fid.py --real_dir {dir_to_real_images}  --gen_dir {dir_to_generated_images}
```

By default, we use MindSpore backend for FID computing (to run inception v3 model inference and extract image features). You may swich to `torchmetrics` backend  by setting `--backend=pt`. The computational difference between these two backends is usually lower than 0.1%, which is neglectable. 

For more usage, please run `python eval/eval_fid.py -h`.

> In the first time running, it will automatically download the checkpoint file for Inception V3 FID model. If you fail to downalod due to network problem, please manually download it from [this link](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/fid/inception_v3_fid-9ec6dfe4.ckpt) and put it under `~/mindspore/models/`

### CLIP Score

Coming soon




## Reference

[1] https://huggingface.co/docs/diffusers/conceptual/evaluation
