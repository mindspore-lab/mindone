#  Stable Video 3D (SV3D)

<p align="center"><img width="600" alt="Output Vis"
src="https://github.com/mindspore-lab/mindone/assets/13991298/0da9cff8-f90a-4fd2-b042-8f92b387a46b"/></p>

## Introduction

Stable Video 3D (SV3D) is a latent video diffusion model for high-resolution, image-to-multi-view generation of orbital videos around a 3D object. It adapts image-to-video diffusion model for novel multi-view synthesis and 3D generation, thereby leveraging the generalization and multi-view consistency of the video models, while further adding explicit camera control for novel view synthesis. An improved 3D optimization techniques was adopted to use SV3D and its NVS outputs for image-to-3D generation.

___Notice that the NeRF part for 3D optimization is not released yet. The current config setup for sv3d_u is based on a finetuned version of SVD with a decoder change.___

<p align="center"><img width="600" alt="SV3D Pipeline Arch"
src="https://github.com/mindspore-lab/mindone/assets/13991298/ac3d6558-e8d6-4636-9298-f2516fcbe63e"/>
<br><em>The 3D generation pipeline leveraging the SV3D multiview diffusion model (for more information please refer to <a href="#acknowledgements">[1]</a>).</em></p>

More demos can be found here. Input images are from [the Unique 3D repo](https://github.com/AiuniAI/Unique3D/tree/main/app/examples).
<details>
<summary>More Demos
</summary>

| Input                                                                                                                | Output                     |
|----------------------------------------------------------------------------------------------------------------------|----------------------------|
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/4f7a0c2f-65c1-4d0a-9861-068b811e0701"/><br/>aaa</p>            | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/ad492ad6-0a7a-4227-8809-b3c8ecf4db65"/><br/>aaa multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/64c269c4-dfee-4495-bede-c7841b137895"/><br/>akun</p>           | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/0588fb26-aa1c-44e0-9b85-e001c6b2e67e"/><br/>akun multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/9655bf80-559c-40bb-8953-c8bdea2d11a3"/><br/>anya</p>           | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/95a15c51-6fa7-4587-8e94-4f979270923f"/><br/>anya multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/8bae9feb-17a1-4cbe-ae56-1f719416e3e8"/><br/>bag</p>            | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/5abff1b5-494f-4321-ae27-6125409515b8"/><br/>bag multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/1b6a650a-d203-461c-a60e-fd03e9434ea8"/><br/>groot</p>          | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/413421b0-79d4-48b3-89a8-13958ff2125d"/><br/>groot multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/5458d1db-807b-4b2e-9f0a-22415f2a0f5e"/><br/>princess-large</p> | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/6bf201a8-da31-4424-8304-42eaf6748501"/><br/>princess-large multiview</p> |

</details>

## Environments
1. To kickstart:
```bash
pip install -r requirements.txt
```
2. Inference is tested on the machine with the following specs using 1x NPU:
    | mindspore |	ascend driver | firmware	| cann toolkit/kernel |
    | :--- | :--- | :--- | :--- |
    | 2.3.1	    | 24.1.RC2 |7.3.0.1.231	| 8.0.RC2.beta1 |

## Pretrained Models
You can easily convert [the SV3D ckpt](https://huggingface.co/stabilityai/sv3d/blob/main/sv3d_u.safetensors) with [our mindone script under svd](https://github.com/mindspore-lab/mindone/blob/master/examples/svd/svd_tools/convert.py).

## Inference

```shell
python simple_video_sample.py \
    --config configs/sv3d_u.yaml \
    --ckpt PATH_TO_CKPT \
    --image PATH_TO_INPUT_IMAGE
```

## Training
1. Prepare the SVD checkpoints as mentioned in the paper. SV3D needs to be finetuned from SVD to cut down training time.
2. Prepare Objaverse overfitting dataset, can refer to our implementation in another 3D project [here](instantmeshpr).
3. The SVD VAE setup is different from the vanilla SV3D structure. To adapt, comment out the VAE setup in the original cfg file, and uncomment those for training. We found that the original cfg setup for SV3D cannot diverge with SVD checkpoints loaded during SV3D training. By modifying the cfgs, the correct VAE can be obtained and overfitting training converges within hours.
```diff

```

4. Launch training by running the following script:
```shell
python train.py --model_cfg configs/sampling/sv3d_u.yaml \
--train_cfg
```

## Acknowledgements

1. Voleti, Vikram, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. "Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion."
