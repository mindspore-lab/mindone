# SadTalker :sob:

A Mindspore implementation of SadTalker based on its [original github](https://github.com/OpenTalker/SadTalker).

## Introduction
SadTalker is a novel system for a stylized audio-driven single image talking head videos animation using the generated realistic 3D motion coefficients (head pose, expression) of the 3DMM.

![sadtalker_1](https://github.com/hqkate/sadtalker/assets/26082447/f1239f9f-f434-4b2c-8ed0-3f07287eb7f3)


## Installation

```bash
pip install -r requirements.txt
```

## Data and Preparation

To execute the inference pipeline of SadTalker, please first download the [pretrained checkpoints](#pretrained-checkpoints) and setup the path in [config file](./config/sadtalker.yaml).

### Pretrained checkpoints

You can download the checkpoints from this link. !!!TODO!!!

After download, the checkpoint folder should be as follow:

<details>
  <summary>data structure: </summary>

    ```bash
    checkpoints/
    ├── BFM_Fitting
    │   ├── 01_MorphableModel.mat
    │   ├── BFM09_model_info.mat
    │   ├── BFM_exp_idx.mat
    │   ├── BFM_front_idx.mat
    │   ├── Exp_Pca.bin
    │   ├── facemodel_info.mat
    │   ├── select_vertex_id.mat
    │   ├── similarity_Lm3D_all.mat
    │   └── std_exp.txt
    ├── ms
    │   ├── ms_audio2exp.ckpt
    │   ├── ms_audio2pose.ckpt
    │   ├── ms_generator.ckpt
    │   ├── ms_he_estimator.ckpt
    │   ├── ms_kp_extractor.ckpt
    │   ├── ms_mapping.ckpt
    │   ├── ms_mapping_full.ckpt
    │   └── ms_net_recon.ckpt
    gfpgan/
    └── weights
        ├── alignment_WFLW_4HG.ckpt
        ├── detection_Resnet50_Final.ckpt
        ├── GFPGANv1.4.ckpt
        └── parsing_parsenet.ckpt
    ```
</details>


### Training Data

We use [VoxCeleb](https://mm.kaist.ac.kr/datasets/voxceleb/) data to train SadTalker. Training codes is still under developement. We will release it when it's ready, thanks!


### Example data for inference

In the original github, there're some example audios and images under [SadTalker/examples](https://github.com/OpenTalker/SadTalker/tree/main/examples). You can download them to quickly start playing Sadtalker! :wink:


## Inference

To generate a talker head video, you have to specify a single portrait image using the argument `--source_image` and an audio file via `--driven_audio`. If you don't specify, it will use the default values.

As reference, you can run the following commands to execute inference process. There're also some arguments to customize the animation, please refer to [input arguments](./utils/arg_parser.py).

```bash
python inference.py --config ./config/sadtalker.yaml --source_image examples/source_image/people_0.png --driven_audio examples/driven_audio/imagine.wav
```

## Examples
1. Chinese


https://github.com/hqkate/sadtalker/assets/26082447/fc20924f-9d42-4432-8f7a-2f8094c23662


2. English

https://github.com/hqkate/sadtalker/assets/26082447/a2ecbf7d-cde4-4fb7-b6d4-6301b679e75b

3. Singing

https://github.com/hqkate/sadtalker/assets/26082447/2c713067-f64e-45a7-9ce2-bc57f340bdad

4. Reference (videos by Pytorch)

- PyTorch with `SynchronizedBatchNorm`:

https://github.com/hqkate/sadtalker/assets/26082447/3de109c8-7231-42c6-9b3d-f150ecd251fa


- PyTorch with `nn.BatchNorm`:

https://github.com/hqkate/sadtalker/assets/26082447/731afe73-e69f-47b4-8e36-6da7be308046


https://github.com/hqkate/sadtalker/assets/26082447/34071866-af30-4520-99d2-fd5a3262f976
