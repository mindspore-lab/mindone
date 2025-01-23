# MVDream
We support the training/inference pipeline of an diffusion-prior based, neural implicit field rendered, 3D mesh generation work called MVDream here.

## Introduction
![intro](https://github.com/user-attachments/assets/2f32333b-f481-4b25-8e43-b4bde1901031)

MVDream is a diffusion model that is able to generate consistent multiview images from a given text prompt. It shows that learning from both 2D and 3D data, a multiview diffusion model can achieve the generalizability of 2D diffusion models and the consistency of 3D renderings.

There are two main folders for the mvdream developments. They are respectively:

* [__MVDream__](https://github.com/bytedance/MVDream): This is an implementation of the t2i sampling pipeline, which can be used as either a t2i inference script or the multiview guidance during the 3D per-scene optimization training.
* [__MVDream-threestudio__](https://github.com/bytedance/MVDream-threestudio): This is an implementation of the 3D per-scene optimization training pipeline, highly modular. With our ms implementation of [threestudio](https://github.com/threestudio-project/threestudio), many other 3D content creation projects listed there can be seamlessly supported.

## Requirements
```bash
pip install -r requirements.txt
```
| mindspore |	ascend driver | firmware	| cann toolkit/kernel |
| :---:     | :---:    | :---:      | :---: |
| 2.4.1	    | 24.1.RC2 | 7.3.0.1.231	| 8.0.RC2.beta1 |

## Inference
The pretrained stable-diffusion-v2.1 model with multiview finetuning will be automatically downloaded out-of-box with the inference t2i script via huggingface, no need to explicitly doing a mannual download. Users are expected to do the _th-ms_ ckpt conversion.
```bash
cd MVDream
python scripts/t2i.py --num_frames 4
python ../../instantmesh/tools/convert_pt2ms.py --src YOUR_HF_PATH/sd-v2.1-base-4view.pt --trgt ./sd-v2.1-base-4view.ckpt  # run t2i again with the converted ckpt
```

You can also finetune your own multiview t2i sd model, theoretically it is mostly similar to the `sv3d_p` training under [SV3D](../sv3d) but with text tokenizer. Our focus in this project is the 3D generation via x0-SDS training rather than finetuning a multiview image/frames/video generation model as in [SV3D](../sv3d).

## Training
We follow the original repo to train the x0-sds optimization pipeline for a given prompt's 3D asset generation for 10k steps.
```bash
cd MVDream-threestudio

# train low res 64x64 with batch_size 8 for 5k steps
python launch.py \
        --train \

# train high res 256x256 with batch_size 4 for another 5k steps
python launch.py \
        --train \
        --train_highres \
        resume="PATH_CKPT_OUTPUT/step_4999.ckpt" \
        system.use_recompute=true \
```
Notice that you need to resume the high-resolution training from the output checkpoint of the low-resolution training. The training happens in a self-supervision manner where the rendered RGB from the renderer is encoded by the guidance pretrained multiview sd-v2.1 model (regarded as "sd2" in the following)'s encoder as a raw latent, and such raw latent is supervised against its own sd2-forward then text-guided-denoised, reconstructed latents.

To generate the output 120 frames rendered from the trained mvdream ckpt, do this:

```bash
python launch.py \
        --test \
        resume="PATH_ABOVE/step9999.ckpt" \
```
The video [here](#training-1) will be generated. Mesh extraction will also be supported.


## Visualization
### Inference
| Input Prompt | Multiview Generation |
| --- | :---:     |
| `an astronaut riding a horse, 3d asset` | ![ms1](https://github.com/user-attachments/assets/a28ef511-71fa-4af7-be0e-97a6c04a23bb) |
| `a DSLR photo of a lion reading the newspaper, 3d asset` | ![ms2](https://github.com/user-attachments/assets/3e8f6c6e-1b91-47c8-87a2-6f29023b5ee2)  |
| `Michelangelo style statue of dog reading news on a cellphone, 3d asset` | ![ms3](https://github.com/user-attachments/assets/77e92964-d9d7-4f76-a63a-8558366bb6e4)   |

### Training
| Input Prompt | 3D Generation |
| --- | :---:     |
| `an astronaut riding a horse` | <video src="https://github.com/user-attachments/assets/f8d00417-96e4-4ddd-aa58-d2c2b7379c8e" /> |

This video is a rendered frame sequence of the generated 3D implicit field by MVDream.

## Performance
Experiments are tested on ascend 910* with mindspore 2.4.1 pynative mode.

### Training
| # samples per ray  | renderer resolution | guidance batch size | speed (frame/second) |
|:---:|:---:|:---:|:---:|
| 64 |64x64 | 8 | 0.857 |
| 64 |256x256 | 4 | 0.465 |

### Inference

| renderer resolution | speed (frame/second) |
|:---------------:|:-------:|
| 256x256 | 0.439 |

## Tips
- **Preview**. Generating 3D content with SDS would a take a lot of time. So The authors suggest to use the 2D multi-view image generation [MVDream](MVDream/README.md) to test if the model can really understand the text before using it for 3D generation.
- **Rescale Factor**. The authors introduce rescale adjustment from [Shanchuan et al.](https://arxiv.org/abs/2305.08891) to alleviate the texture over-saturation from large CFG guidance. However, in some cases, the authors find it to cause floating noises in the generated scene and consequently OOM issue. Therefore the authors reduce the rescale factor from 0.7 in original paper to 0.5. However, if you still encounter such a problem, please try to further reduce `system.guidance.recon_std_rescale=0.3`.

## Acknowledgements
1. Shi, Yichun, et al. "MVDream: Multi-view Diffusion for 3D Generation." The Twelfth International Conference on Learning Representations.
