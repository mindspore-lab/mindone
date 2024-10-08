# Mindspore Implementation of Marigold

This repository is the mindspore implementation of Marigold, which paper titled "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" and is accepted by CVPR 2024. The code is based on [official implementation](https://github.com/prs-eth/Marigold).

![demo](doc/demo.png)

## Dependencies

The train and inference code was tested on:

- **1*ascend-snt9b|ARM: 24核 192GB**
- **python3.9, mindspore-2.2.14, cann7.0.0.beta1**

The other dependent libraries has been recorded in **requirements.txt**, please first install above package and `mindone`, then use command below to install the environment.

```bash
pip install -r requirements.txt
```

## Checkpoint Download

The official release [checkpoint](https://huggingface.co/prs-eth/marigold-v1-0) is stored in the Hugging Face cache and [official storage](https://share.phys.ethz.ch/~pf/bingkedata/marigold/checkpoint/marigold-v1-0.tar).

You can use the following script to download the checkpoint weights locally or access Hugging Face cache to download:

```bash
bash script/download_weights.sh marigold-v1-0
```

If you can't access hugging face or official storage, you can use following script to download the checkpoint I have upload to [mindspore platform](https://xihe.mindspore.cn/models/Braval/Marigold-Model). (Please make sure you have `git-lfs` installed before running, if not, you could reference [git-lfs](https://github.com/git-lfs/git-lfs) to install it.)

```bash
bash script/download_weights_mindspore.sh
```

After download, the checkpoint path should be like this:

```
marigold
|——marigold-checkpoint
|   |——marigold-v1-0
...
```

We recommend to run inference with this checkpoint.

## Testing on your images <a name="infer"></a>

### Prepare images

1. Use selected images from official paper:

    ```bash
    bash script/download_sample_data.sh
    ```

   If you can't access the official storage, you can use following script to download the sample images I have upload to [mindspore platform](https://xihe.mindspore.cn/datasets/Braval/Marigold-Example).

    ```bash
    bash script/download_sample_data_mindspore.sh
    ```

2. Or place your images in the directory `input/in-the-wild_example`, and run the following inference command.

    ```bash
    python run.py --fp16
    ```

    It will infer your images with default settings as below:

    ```bash
    python run.py \
        --fp16 \
        --checkpoint marigold-checkpoint/marigold-v1-0 \
        --denoise_steps 50 \
        --ensemble_size 2 \
        --input_rgb_dir input/in-the-wild_example \
        --output_dir output/in-the-wild_example
    ```

    Then you can find all results in `output/in-the-wild_example`.

### Inference settings

The default settings are optimized for the best result. However, the behavior of the code can be customized:

- Trade-offs between the **accuracy** and **speed** (for both options, larger values result in better accuracy at the cost of slower inference.)
  - `--ensemble_size`: Number of inference passes in the ensemble. Default: 2 (for run.py) 、10 (for infer.py).
  - `--denoise_steps`: Number of denoising steps of each inference pass. For the original (DDIM) version, it's recommended to use 10-50 steps. When unassigned (`None`), will read default setting from config. Default: 50 (for run.py and infer.py).

- By default, the inference script resizes input images to the *processing resolution*, and then resizes the prediction back to the original resolution. This gives the best quality, as Stable Diffusion, from which Marigold is derived, performs best at 768x768 resolution.  

  - `--processing_res`: the processing resolution; set as 0 to process the input resolution directly. When unassigned (`None`), will read default setting from model config. Default: 768.
  - `--output_processing_res`: produce output at the processing resolution instead of upsampling it to the input resolution. Default: False.
  - `--resample_method`: the resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic`, or `nearest`. Default: `bilinear`.

- `--half_precision` or `--fp16`: Run with half-precision (16-bit float) to reduce memory usage and faster, which might lead to suboptimal results.
- `--color_map`: [Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) used to colorize the depth prediction. Default: Spectral. Set to `None` to skip colored depth map generation.

## Evaluation on test datasets <a name="evaluation"></a>

First download [evaluation datasets](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset) into corresponding subfolders using command below:

```bash
bash script/download_eval_data.sh
```

If you can't access the official storage, you can use following script to download the evaluation datasets NYUv2 and KITTI, which I have upload to [mindspore platform](https://xihe.mindspore.cn/datasets/Braval/Marigold-Eval).

```bash
bash script/download_eval_data_mindspore.sh
```

After download, the datasets path should be like this:

```
marigold
|——marigold-data
|   |——nyuv2
|   |   ㇗nyu_labeled_extracted.tar
|   |——kitti
|   |   ㇗kitti_eigen_split_test.tar
...
```

Run inference and evaluation scripts seperately, for example:

```bash
# Run inference
bash script/eval/11_infer_nyu.sh --fp16

# Evaluate predictions
bash script/eval/12_eval_nyu.sh
```

**Note**: Although the seed has been set, the results might still be slightly different on different hardware.

## Training

As for official training dataset Hypersim is too big, the code is only verified for training on dataset [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/).

You can download offical rgb and depth zip of Virtual KITTI, and run scripts as follow to get needed form for train.

```bash
bash script/download_vkitti.sh
```

If you can't access the official VKITTI, you can use following script to download the dataset I have upload to [mindspore platform](https://xihe.mindspore.cn/models/Braval/Marigold-Model).

```bash
bash script/download_vkitti_mindspore.sh
```

After download, the train datasets path should be like this:

```
marigold
|——marigold-data
|   |——vkitti
|   |   ㇗vkitti.tar
...
```

It's recommended to train in graph mode, to acquire faster performance. Please download mindspore implementation of Stable Diffusion v2 [checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt) first, and save it into `marigold-checkpoint` like following command:

```bash
mkdir -p marigold-checkpoint
wget -nv --show-progress -P marigold-checkpoint https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt
```

Then please run the command below to train your own depth estimation diffusion model in graph mode:

```bash
python train_graph.py \
    --train_config "config/train_marigold.yaml" \
    --output_path "output/graph-train" \
    --pretrained_model_path "marigold-checkpoint/sd_v2_768_v-e12e3a9b.ckpt"
```

Resume train is not supported yet. It's recommended to train without interruption.

### Evaluating results

To use your own training result to infer, please first move it to *marigold-checkpoint* and rename as **marigold-vkitti.ckpt**, which looks like this:

```
marigold
|——marigold-checkpoint
|   |——marigold-vkitti.ckpt
...
```

And then you could orgnized your data as [infer](#infer) and [evaluation](#evaluation) has introduced. And run below command to infer on your own ckpt:

```bash
python run.py --fp16 --checkpoint marigold-checkpoint/marigold-vkitti.ckpt --ms_ckpt
```

Or run following command to eval on datasets.

```bash
# Run inference
bash script/eval/11_infer_nyu_mindspore.sh --fp16

# Evaluate predictions
bash script/eval/12_eval_nyu.sh
```

**Note**: The training code is still being updated to ensure the training result which not performs good now.

## Citation

Thanks to the official [Marigold](https://github.com/prs-eth/Marigold) repository. And cite their paper here:

```bibtex
@InProceedings{ke2023repurposing,
      title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
      author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024}
}
```

## License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
