# InstantMesh: 3D Mesh Generation from Multiview Images

This `instantmesh` module under `.../sv3d/models` is implemented for the 3D mesh generation using the multiview images extracted from [the sv3d pipeline](../../simple_video_sample.py).

A walk-through of the files in this module is as follows.

<details>
<summary>Files Tree
</summary>

```bash
├── instantmesh
│   ├── geometry                # use Flexicubes to extract isosurface
│   │   ├── rep_3d
│   │   │   ├── flexicubes_geometry.py
│   │   │   ├── tables.py
│   │   │   └── flexicubes.py
│   │   ├── render
│   │   │   ├── neural_render.py
│   │   └── camera
│   │       └── perspective_camera.py
│   ├── decoder                 # triplane feature transformer decoder
│   │   └── transformer.py
│   ├── encoder                 # dino vit decoder to extract img feat
│   │   ├── dino_wrapper.py
│   │   ├── dino.py
│   ├── renderer                # a wrapper that synthesizes sdf/texture from triplane feat
│   │   ├── synthesizer_mesh.py
│   │   ├── utils
│   │   │   └── renderer.py
│   │   └── synthesizer.py
│   └── lrm_mesh.py
├── utils
│   ├── camera_util.py
│   ├── train_util.py
│   └── mesh_util.py
└── configs
    └── instant-mesh-large.yaml
```

</details>

## Introduction

InstantMesh [[1]](#acknowledgements) synergizes the strengths of a multiview diffusion model and a sparse-view reconstruction model based on the LRM [[2]](#acknowledgements) architecture. It also adopts FlexiCubes [[3]](#acknowledgements) isosurface extraction for a smoother and more elegant mesh extraction.

Using the multiview images input from 3D mesh extracted from [the sv3d pipeline](../../simple_video_sample.py), we extracted 3D meshes as below. Please kindly find the input illustrated by following the link to the sv3d pipeline above.

| <p align="center"> akun </p>                                                                                                                                                                                                                                                                                                                                                                          | <p align="center"> anya </p>                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <div class="sketchfab-embed-wrapper"><iframe title="akun_ms" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/c8b5b475529d48589b85746aab638d2b/embed"></iframe></div> | <div class="sketchfab-embed-wrapper"><iframe title="anya_ms" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/180fd247ba2f4437ac665114a4cd4dca/embed"></iframe></div> |

The illustrations here are better viewed in viewers than with HTML support (e.g., the vscode built-in viewer).

## Environments

1. To kickstart:

```bash
pip install -r requirements.txt
```

2. Inference is tested on the machine with the following specs using 1x NPU:

```text
Architecture:                    aarch64
CPU op-mode(s):                  64-bit
Byte Order:                      Little Endian
CPU(s):                          192
Thread(s) per core:              1
Core(s) per socket:              48
Vendor ID:                       HiSilicon
Model name:                      Kunpeng-920
NPU:                             910B1
Mindspore Version:               2.3.0.B528
CANN Version:                    7.3
Ascend Driver:                   23.0.rc3.6
```

## Pretrained Models

To better accommodate the mindone transformer codebase, we provide an out-of-the-box [checkpoints conversion script](../../tools/convert_instantmesh_ckpt.py) that works seamlessly with the mindspore version of transformers.

The image features are extracted with dino-vit, which depends on HuggingFace's transformer package. We reuse [the MindSpore's implementation](https://github.com/mindspore-lab/mindone/blob/master/mindone/transformers/modeling_utils.py#L499) and the only challenge remains to be that `.bin` checkpoint of [dino-vit](https://huggingface.co/facebook/dino-vitb16/tree/main) is not supported by MindSpore off-the-shelf. The checkpoint script above serves easy conversion purposes and ensures that dino-vit is still based on `MSPreTrainedModel` safe and sound.

## Inference

```shell
python mv2mesh_instantmesh.py --ckpt PATH_TO_CKPT \
--input_vid PATH_TO_INPUT_MULTIVIEW_VID
```

## Acknowledgements

1. Shen, Tianchang, et al. "Flexible Isosurface Extraction for Gradient-Based Mesh Optimization." ACM Trans. Graph. 42.4 (2023): 37-1.
2. Hong, Yicong, et al. "Lrm: Large reconstruction model for single image to 3d." arXiv preprint arXiv:2311.04400 (2023).
3. Shen, Tianchang, et al. "Flexible Isosurface Extraction for Gradient-Based Mesh Optimization." ACM Trans. Graph. 42.4 (2023): 37-1.
4. Lorensen, William E., and Harvey E. Cline. "Marching cubes: A high resolution 3D surface construction algorithm." Seminal graphics: pioneering efforts that shaped the field. 1998. 347-353.
