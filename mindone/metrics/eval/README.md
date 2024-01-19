
This folder contains code for diffusion model evaluation, e.g.,

- Fréchet video distance (FVD)
- Kernel video distance (FVD)


Note that all the above metrics are computed based on neural network models.

> A convincing evaluation for diffusion models requires both visually qualitative comparision and quantitative measure. A lower FVD score does not nessarily show one model is better than another.

## Usage


### FVD and KVD

FVD and KVD measure how close a distribution of generated videos is to the ground-truth distribution. To compute the FVD and KVD between the real videos and the generated videos, please run

```
python mindone/metrics/eval/eval_fvd_kvd.py --real_video_dir {dir_to_real_videos}  --gen_video_dir {dir_to_generated_videos}
```

For more usage, please run `python mindone/metrics/eval/eval_fvd_kvd.py -h`.

> The default ckpt is None, it will automatically download the InceptionI3d checkpoint file for FVD and KVD. If you fail to downalod due to network problem, please manually download it from [this link](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/fvd/inception_i3-02b0bb54.ckpt).


## Reference

[1] https://github.com/YingqingHe/LVDM/tree/main
