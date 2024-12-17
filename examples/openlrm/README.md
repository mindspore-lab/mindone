# OpenLRM: Open-Source Large Reconstruction Models
This is the mindspore re-implementation of the work [![LRM](https://img.shields.io/badge/LRM-Original%20Paper-green)](https://arxiv.org/abs/2311.04400) based on open source repo [OpenLRM](https://github.com/3DTopia/OpenLRM).


<img src="assets/rendered_video/teaser.gif" width="75%" height="auto"/>

<div style="text-align: left">
    <img src="assets/mesh_snapshot/crop.owl.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.owl.ply01.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.building.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.building.ply01.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.rose.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.rose.ply01.png" width="12%" height="auto"/>
</div>

## **Introduction**
LRM is the first Large Reconstruction Model that predicts the 3D model of an object from a single input image within just few seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. LRM model was trained in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs, including real-world in-the-wild captures and images created by generative models. 

## Get Started
### Requirements
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.3.1 | 24.1RC2 | 7.3.0.1.231 | 8.0.RC2.beta1|

### Dependencies
- Install requirements for OpenLRM first.
  ```
  pip install -r requirements.txt
  ```


<!-- ### Quick Start -->

### Pretrained Models

- Model weights are released on [Hugging Face](https://huggingface.co/zxhezexin).
- Weights will be downloaded automatically when you run the inference script for the first time.
- Please be aware of the [license](LICENSE_WEIGHT) before using the weights.

| Model | Training Data | Layers | Feat. Dim | Trip. Dim. | In. Res. | Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| openlrm-obj-small-1.1 | Objaverse | 12 | 512 | 32 | 224 | [HF](https://huggingface.co/zxhezexin/openlrm-obj-small-1.1) |
| openlrm-obj-base-1.1 | Objaverse | 12 | 768 | 48 | 336 | [HF](https://huggingface.co/zxhezexin/openlrm-obj-base-1.1) |
| openlrm-obj-large-1.1 | Objaverse | 16 | 1024 | 80 | 448 | [HF](https://huggingface.co/zxhezexin/openlrm-obj-large-1.1) |
| openlrm-mix-small-1.1 | Objaverse + MVImgNet | 12 | 512 | 32 | 224 | [HF](https://huggingface.co/zxhezexin/openlrm-mix-small-1.1) |
| openlrm-mix-base-1.1 | Objaverse + MVImgNet | 12 | 768 | 48 | 336 | [HF](https://huggingface.co/zxhezexin/openlrm-mix-base-1.1) |
| openlrm-mix-large-1.1 | Objaverse + MVImgNet | 16 | 1024 | 80 | 448 | [HF](https://huggingface.co/zxhezexin/openlrm-mix-large-1.1) |

Model cards with additional details can be found in [model_card.md](model_card.md).

### Prepare Images
- We put some sample inputs under `assets/sample_input`, and you can quickly try them.
- Prepare RGBA images or RGB images with white background (with some background removal tools, e.g., [Rembg](https://github.com/danielgatis/rembg), [Clipdrop](https://clipdrop.co)).

### Inference
- Run the inference script to get 3D assets.
- You may specify which form of output to generate by setting the flags `EXPORT_VIDEO=true` and `EXPORT_MESH=true`.
- Please set default `INFER_CONFIG` according to the model you want to use. E.g., `infer-b.yaml` for base models and `infer-s.yaml` for small models.
- An example usage is as follows:

  ```
  # Example usage
  EXPORT_VIDEO=true
  EXPORT_MESH=true
  INFER_CONFIG="./configs/infer-b.yaml"
  MODEL_NAME="zxhezexin/openlrm-mix-base-1.1"
  IMAGE_INPUT="./assets/sample_input/owl.png"

  python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
  ```

<!-- ### Tips
- The recommended PyTorch version is `>=2.1`. Code is developed and tested under PyTorch `2.1.2`.
- If you encounter CUDA OOM issues, please try to reduce the `frame_size` in the inference configs.
- You should be able to see `UserWarning: xFormers is available` if `xFormers` is actually working. -->

## Training

### Configuration
- We provide a sample accelerate config file under `configs/accelerate-train.yaml`, which defaults to use 8 GPUs with `bf16` mixed precision.
- You may modify the configuration file to fit your own environment.

### Data Preparation
- We provide the core [Blender script](scripts/data/objaverse/blender_script.py) used to render Objaverse images.
- Please refer to [Objaverse Rendering](https://github.com/allenai/objaverse-rendering) for other scripts including distributed rendering.

### Run Training
- A sample training config file is provided under `configs/train-sample.yaml`.
- Please replace data related paths in the config file with your own paths and customize the training settings.
- An example training usage is as follows:

  ```
  # Example usage
  ACC_CONFIG="./configs/accelerate-train.yaml"
  TRAIN_CONFIG="./configs/train-sample.yaml"

  accelerate launch --config_file $ACC_CONFIG -m openlrm.launch train.lrm --config $TRAIN_CONFIG
  ```

### Inference on Trained Models
- The inference pipeline is compatible with huggingface utilities for better convenience.
- You need to convert the training checkpoint to inference models by running the following script.

  ```
  python scripts/convert_hf.py --config <YOUR_EXACT_TRAINING_CONFIG> convert.global_step=null
  ```

- The converted model will be saved under `exps/releases` by default and can be used for inference following the [inference guide](https://github.com/3DTopia/OpenLRM?tab=readme-ov-file#inference).


## License

- OpenLRM as a whole is licensed under the [Apache License, Version 2.0](LICENSE), while certain components are covered by [NVIDIA's proprietary license](LICENSE_NVIDIA). Users are responsible for complying with the respective licensing terms of each component.
- Model weights are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE_WEIGHT). They are provided for research purposes only, and CANNOT be used commercially. 
