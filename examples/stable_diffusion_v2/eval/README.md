
This folder contains code and scripts for diffusion model evaluation, including 

- Fréchet inception distance (FID)
- CLIP score
- CLIP directional similarity 

Note that all these metrics are neural network-based thus the parameters used are kept the same as the pytorch version used in other benchmarks.

## Usage

### FID

To compute the FID between the real images and the generated images, please run 

```
python eval/eval_fid.py --real_dir {dir_to_real_images}  --gen_dir {dir_to_generated_images}
```

By default, we use the mindspore backend to run inception v3 model inference for FID computing. You may swich to `torchmetrics` backend  by setting `--backend=pt`. The computational difference between these two backend is usually lower than 0.1%, which is neglectable. 


### CLIP Score

Coming soon


## Reference

[1] https://huggingface.co/docs/diffusers/conceptual/evaluation
