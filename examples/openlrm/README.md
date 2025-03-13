# OpenLRM: Open-Source Large Reconstruction Models
This is the mindspore re-implementation of the work [LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) based on open source repo [OpenLRM v1.1.1](https://github.com/3DTopia/OpenLRM).

## Introduction
<p align="center">
  <img src="https://github.com/user-attachments/assets/4f5d4325-1fd5-4421-a958-9a11ecee0192"  width="70%" height="auto">
</p>
LRM is the first Large Reconstruction Model that predicts the 3D model of an object from a single input image within just few seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. LRM model was trained in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs, including real-world in-the-wild captures and images created by generative models.

## Get Started
### Requirements
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.4.1 | 24.1RC2 | 7.3.0.1.231 | 8.0.RC3.beta1|

### Dependencies
- Install requirements for OpenLRM.
  ```
  pip install -r requirements.txt
  ```

## Inference

### Prepare Images
- Prepare RGBA images or RGB images with white background (with some background removal tools, e.g., [Rembg](https://github.com/danielgatis/rembg), [Clipdrop](https://clipdrop.co)).

### Pretrained Models

- Model weights are released on [Hugging Face](https://huggingface.co/zxhezexin) with different [configurations]().
- We suggest to download checkpoints and configs locally, though weights will be downloaded automatically when you run the inference script for the first time.
  ```
  weight_dir
  ├── config.json
  └── model.safetensors
  ```

- Please be aware of the [license](LICENSE_WEIGHT) before using the weights.

### Run Inference on Pretrained Models
- Run the inference script for image-to-3D.
- You may specify which form of output to generate by setting the flags `EXPORT_VIDEO=true` and `EXPORT_MESH=true`.
- Please set default `INFER_CONFIG` according to the model you want to use: `infer-s.yaml` for small models, `infer-b.yaml` for base models and `infer-l.yaml` for large models.
- A sample script is provided in `scripts/infer_hf.sh`. An example usage is as follows:

  ```
  # Example usage
  EXPORT_VIDEO=true
  EXPORT_MESH=true
  INFER_CONFIG="./configs/infer-b.yaml"
  MODEL_NAME="zxhezexin/openlrm-mix-base-1.1"
  IMAGE_INPUT="./assets/sample_input/owl.png"

  python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
  ```


## Training

### Configuration
- We provide training sample config files under `configs/train-sample-X.yaml`, which defaults to use 1 NPU with `fp32` precision in PYNATIVE_MODE.
- `configs/train-sample-large.yaml` can use 1 NPU with `bf16` mixed precision for training.
- Currently we only support loading Objaverse data curated by script in `scripts/data/objaverse/blender_script.py`.

  | Model | Dataset | Layers | Feat. Dim | Trip. Dim. | In. Res. |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | small | Objaverse | 12 | 512 | 32 | 224 |
  | base | Objaverse | 12 | 768 | 48 | 336 |
  | large | Objaverse | 12* | 768* | 80 | 448 |

  *`large` config using a different transformer architecture from original OpenLRM to support training with 1 card with 65GB.

### Data Preparation
- We provide the core [Blender script](scripts/data/objaverse/blender_script.py) used to render Objaverse images.
- Please refer to [Objaverse Rendering](https://github.com/allenai/objaverse-rendering) for other scripts including distributed rendering.
- Objaverse dataset format:

  ```
  objaverse_dataset
  ├── meta.json
  ├── <UID#1>
  │   ├── pose
  │   │   ├── 000.npy
  │   │   ├── 001.npy
  │   │   ├── 002.npy
  │   │   ├── ...
  │   │   └── 031.npy
  │   ├── rgba
  │   │   ├── 000.png
  │   │   ├── 001.png
  │   │   ├── 002.png
  │   │   ├── ...
  │   │   └── 031.png
  │   └── intrinsics.npy
  ├── <UID#2>
  │   ├── pose
  │   ├── rgba
  │   └── intrinsics.npy
  ├── ...
  └── <UID#N>
      ├── pose
      ├── rgba
      └── intrinsics.npy
  ```

### Run Training
- Sample training config files are provided under `configs/train-sample-X.yaml`.
- Please replace data related paths in the config file with your own paths and customize the training settings.
- Some training sample scripts are provided under `scripts/train_X.sh`. An example training usage is as follows:

  ```
  # Example usage
  export MS_INDEPENDENT_DATASET=True
  export MS_ALLOC_CONF="enable_vmm:True;"
  export TRAIN_CONFIG="./config/train-sample-small.yaml"
  export DEVICE_ID=0
  EPOCH=100000
  OUTPUT_PATH=./outputs/train_small_e$EPOCH
  python -m openlrm.launch train.lrm --config $TRAIN_CONFIG --mode 1 --num_parallel_workers 1 --epochs $EPOCH
  ```

### Run Inference on Trained MindSpore Models
- A inference sample script is provided in `scripts/infer_ms.sh` to infer 3D assets and render videos by trained MindSpore checkpoints.
- An example inference usage is as follows:

  ```
  # Example usage
  EXPORT_VIDEO=true
  EXPORT_MESH=true
  INFER_CONFIG="./configs/infer-s.yaml"
  MODEL_NAME="outputs/small/2024-12-30T18-24-53"
  IMAGE_INPUT="sample_input"

  epoch=100000
  MODEL_CKPT=openlrm-e${epoch}.ckpt
  DEVICE_ID=0 python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME model_ckpt=$MODEL_CKPT epoch=$epoch image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
  ```
## Evaluation
### Evaluation Data Preparation
- Same format as [Data Preparation](#data-preparation).

### Run Evaluation on Trained Checkpoint
- Evaluation script `eval.py` input first image and camera information as input, and based on other camera information output other views of 3D model.
- Set evaluation data path in `MODEL_PATH/cfg.yaml`:
  ```
  dataset:
      subsets:
          -   name: objaverse
              meta_path:
                  val: <REPLACE_WITH_VAL_UIDS_IN_JSON>  # e.g. val_dataset/meta.json
  ```
- Compute and report inference speed, PSNR and SSIM at `OUTPUT_PATH/EPOCH/log_0.txt`.
- A sample evalution script is in `scripts/eval.sh`. An example is as follows:
  ```
  # Example usage
  MODEL_PATH=outputs/base/2025-01-06T14-37-07
  CKPT_NAME=openlrm-e50000.ckpt
  OUTPUT_PATH=output/base
  DEVICE_ID=0 python eval.py --model_path $MODEL_PATH --ckpt_name $CKPT_NAME --output_path $OUTPUT_PATH
  ```

## Performace

### Inference Performance
Experiments are tested on ascend 910* with mindspore 2.4.1 pynative mode.
- Input a single image, here reports speed of image-to-3D and image rendering.

| model name|  cards| batch size | resolution | precision | flash attn| (image to 3D) s/step | (render) img/s|  recipe| weight|
|---|---|---|---|---|---|---|---|---|---|
|openlrm-mix-base-1.1 |  1 | 1 | 336x336 |fp32| OFF |   0.83 | 0.98 |[yaml](./configs/infer-b.yaml)| [weight](https://huggingface.co/zxhezexin/openlrm-mix-base-1.1)|
|openlrm-obj-small-1.1 |  1 | 1 | 224x224 |fp32| OFF |  0.72  | 2.19 |[yaml](./configs/infer-s.yaml)| [weight](https://huggingface.co/zxhezexin/openlrm-obj-small-1.1)|
|openlrm-obj-base-1.1 |  1 | 1 | 336x336 |fp32| OFF | 0.76   | 0.98 |[yaml](./configs/infer-b.yaml)| [weight](https://huggingface.co/zxhezexin/openlrm-obj-base-1.1)|
|openlrm-obj-large-1.1 |  1 | 1 | 448x448 |fp32| OFF | 0.92  | 0.37|[yaml](./configs/infer-l.yaml)| [weight](https://huggingface.co/zxhezexin/openlrm-obj-large-1.1)|

|Input| Output |
|:---|:---|
|<img src="https://github.com/user-attachments/assets/baf30c16-384a-44fc-b303-ade6d7a3b8b0"  width="256px"> | <img src="https://github.com/user-attachments/assets/bcb6d8a1-b3f2-41a7-86a0-c222de6b0987"  width="256px">  |
| <img src="https://github.com/user-attachments/assets/dc96aa17-35cd-4477-b62d-8ba11dd6ba47"  width="256px"> | <img src="https://github.com/user-attachments/assets/ab2da9bb-7fa4-4bc4-9e84-a151b41b7e36"  width="256px"> |
|  <img src="https://github.com/user-attachments/assets/f4f08604-f5d1-46cb-b69b-d09b41cbad58" width="256px">|  <img src="https://github.com/user-attachments/assets/6362ae00-5a17-4ae6-909b-7777c2aaca0d" width="256px"> |
| <img src="https://github.com/user-attachments/assets/e6cdcf8e-06d1-4890-b78a-908b78d10437" width="256px">| <img src="https://github.com/user-attachments/assets/bf5a8aec-79e6-4429-8f15-d82390d41321"  width="256px"> |


### Training Performance
Experiments are tested on ascend 910* with mindspore 2.4.1 pynative mode.
- Train with objaverse data only:

|model name	| cards	| batch size	| resolution	| precision| recompute	|  s/step	| batch/s|
|---|---|---|---|---|---|---|---|
|small | 1 | 1 | (4x)224x224 | fp32 | OFF |  2.00 | 0.50 |
|base | 1 | 1 | (4x)336x336 | fp32 | OFF |  2.63 | 0.38 |
|large | 1 | 1 | (4x)448x448 | bf16 | ON |  3.12 | 0.32|  


- Evaluation on trained models:

|model name	| epoch| cards	| batch size	| resolution	| precision|  (infer+render) s/step	| PSNR | SSIM |
|---|---|---|---|---|---|---|---|---|
|small | 100k |1 | 1 | 224x224 | fp32  | 0.99 | 26.38 | 0.91  
|base | 100k |1 | 1 | 336x336 | fp32  | 1.64 | 27.44 | 0.92  
|large | 50k |1 | 1 | 448x448 | bf16  | 2.87 | 22.98 | 0.89  

|Input| Output (small-ep50k)| Output (base-ep40k) |
|:---|:---|:---|
|<img src="https://github.com/user-attachments/assets/102e3a5e-1d78-4c73-8eb7-658ca83fdf3f"  width="256px"> |<img src="https://github.com/user-attachments/assets/88ed101d-6ae6-4192-972f-3c89c5a8c840"  width="256px"> |<img src="https://github.com/user-attachments/assets/50304993-8e59-4419-8084-d1146f4d28c9" width="256px"> |
|<img src="https://github.com/user-attachments/assets/05c2d0e6-40bd-4be2-8e51-21d01d72c456"  width="256px"> |<img src="https://github.com/user-attachments/assets/be5d3d0a-2d2e-498e-8308-a90b6ef850d4"  width="256px"> |<img src="https://github.com/user-attachments/assets/cd3b4c41-2f57-440d-9b6e-6ce35ee883b2"  width="256px"> |
|<img src="https://github.com/user-attachments/assets/803e0671-7f29-4467-a9d4-04211fd35b35"  width="256px">|<img src="https://github.com/user-attachments/assets/64189b53-a4a7-4590-b3b5-d30f653f855d" width="256px"> |<img src="https://github.com/user-attachments/assets/ea9d3404-0538-4ce0-ab72-06923e6f1504" width="256px"> |
|<img src="https://github.com/user-attachments/assets/a8c870b2-b616-415f-829d-a2ab3e581d62"  width="256px"> |<img src="https://github.com/user-attachments/assets/fd023b2b-058b-444d-93d5-3a3e037a3918"  width="256px"> |<img src="https://github.com/user-attachments/assets/fcdcde93-c382-464a-9faa-5bf6350e2bbf"  width="256px">  |

## License

- OpenLRM as a whole is licensed under the [Apache License, Version 2.0](LICENSE), while certain components are covered by [NVIDIA's proprietary license](LICENSE_NVIDIA). Users are responsible for complying with the respective licensing terms of each component.
- Model weights are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE_WEIGHT). They are provided for research purposes only, and CANNOT be used commercially.
