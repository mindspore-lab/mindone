# InstantMesh: 3D Mesh Generation from Multiview Images

We support [instantmesh](https://github.com/TencentARC/InstantMesh) for the 3D mesh generation using the multiview images extracted from [the sv3d pipeline](https://github.com/mindspore-lab/mindone/pull/574).
<p align="center" width="100%">
  <img width="746" alt="Capture" src="https://github.com/user-attachments/assets/be5cf033-8f89-4cad-97dc-2bf76c1b7a4d">
</p>

The model consists of a Dino-ViT feature extractor, a triplane feature extraction transformer, and a triplane-to-NeRF synthesizer which also conducts rendering.

A walk-through of the file structure is provided here as below.

<details>
<summary>Files Tree
</summary>

```bash
├── models
│   ├── decoder                 # triplane feature transformer decoder
│   │   └── transformer.py
│   ├── encoder                 # dino vit decoder to extract img feat
│   │   ├── dino_wrapper.py
│   │   └── dino.py
│   ├── renderer                # a wrapper that synthesizes sdf/texture from triplane feat
│   │   ├── synthesizer_mesh.py # triplane synthesizer, the triplane feat is decoded thru nerf to predict texture rgb & 3D sdf
│   │   ├── synthesizer.py      # triplane synthesizer, the triplane feat is decoded thru nerf to predict novel view rgba
│   │   └── utils
│   │       └── renderer.py
│   ├── geometry                # use Flexicubes to extract isosurface
│   │   ├── rep_3d
│   │   │   ├── flexicubes_geometry.py
│   │   │   ├── tables.py
│   │   │   └── flexicubes.py
│   │   └── camera
│   │       └── perspective_camera.py
│   ├── lrm_mesh.py             # model arch for the instantmesh inference
│   └── lrm.py                  # model arch for the instantmesh stage 1 training
├── utils
│   ├── camera_util.py
│   ├── train_util.py
│   ├── eval_util.py
│   ├── loss_util.py
│   ├── ms_callback_util.py
│   └── mesh_util.py
├── data
│   └── objaverse.py            # training dataset definition and batchify
├── configs
│   └── instant-mesh-large.yaml
├── inference.py                # instantmesh inference
├── train.py                    # instantmesh stage 1 training
├── eval.py                     # instantmesh stage 1 evaluation, mview imgs to novel view synthesis
└── model_stage1.py             # model arch for the stage 1 training
```

</details>

## Introduction

InstantMesh [[1]](#acknowledgements) synergizes the strengths of a multiview diffusion model and a sparse-view reconstruction model based on the LRM [[2]](#acknowledgements) architecture. It also adopts FlexiCubes [[3]](#acknowledgements) isosurface extraction for a smoother and more elegant mesh extraction.

Using the multiview images input from 3D mesh extracted from [the sv3d pipeline](../../simple_video_sample.py), we extracted 3D meshes as below. Please kindly find the input illustrated by following the link to the sv3d pipeline above.

| <p align="center"> akun </p>                                                                                                                                                                                                                                                                                                                                                                          | <p align="center"> anya </p>                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <div class="sketchfab-embed-wrapper"><iframe title="akun_ms" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/c8b5b475529d48589b85746aab638d2b/embed"></iframe></div> | <div class="sketchfab-embed-wrapper"><iframe title="anya_ms" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/180fd247ba2f4437ac665114a4cd4dca/embed"></iframe></div> |

The illustrations here are better viewed in viewers than with HTML support (e.g., the vscode built-in viewer).

## Environment Requirements

1. To kickstart:

```bash
pip install -r requirements.txt
```

2. Inference is tested on the machine with the following specs using 1x NPU:

| mindspore |	ascend driver | firmware	| cann toolkit/kernel |
| :--- | :--- | :--- | :--- |
| 2.3.1	    | 24.1.RC2 |7.3.0.1.231	| 8.0.RC2.beta1 |

## Pretrained Models
### ViT Pretrained Checkpoint
To better accommodate the mindone transformer codebase, we provide an out-of-the-box [checkpoints conversion script](./tools/convert_dinovit_bin2st.py) that works seamlessly with the mindspore version of transformers.

The image features are extracted with dino-vit, which depends on HuggingFace's transformer package. We reuse [the MindSpore's implementation](https://github.com/mindspore-lab/mindone/blob/master/mindone/transformers/modeling_utils.py#L499) and the only challenge remains to be that `.bin` checkpoint of [dino-vit](https://huggingface.co/facebook/dino-vitb16/tree/main) is not supported by MindSpore off-the-shelf. The checkpoint script above serves easy conversion purposes and ensures that dino-vit is still based on `MSPreTrainedModel` safe and sound.

### InstantMesh Checkpoint
To convert checkpoints, we prepare the following snippet.
```bash
python tools/convert_pt2ms.py --trgt PATH_TO_CKPT
```

## Inference

```shell
python inference.py --ckpt PATH_TO_CKPT \
--input_vid PATH_TO_INPUT_MULTIVIEW_VID
```

## Training
```shell
python train.py --base configs/YOUR_CFG
```
One needs to patch `mindcv.models.vgg` in L62 to enable conv kernel bias to align with the torchmetric implementation of lpips loss.
```diff
- conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode="pad", padding=1)
+ conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode="pad", padding=1, has_bias=True)
```

### Data Curation
We used Blender to render multiview frames for a 3D object in `.obj` for training.

## Acknowledgements

1. Xu, Jiale, et al. "Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models." arXiv preprint arXiv:2404.07191 (2024).
2. Hong, Yicong, et al. "Lrm: Large reconstruction model for single image to 3d." arXiv preprint arXiv:2311.04400 (2023).
3. Shen, Tianchang, et al. "Flexible Isosurface Extraction for Gradient-Based Mesh Optimization." ACM Trans. Graph. 42.4 (2023): 37-1.
4. Lorensen, William E., and Harvey E. Cline. "Marching cubes: A high resolution 3D surface construction algorithm." Seminal graphics: pioneering efforts that shaped the field. 1998. 347-353.
