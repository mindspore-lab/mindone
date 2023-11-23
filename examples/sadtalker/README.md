# SadTalker :cry:

A Mindspore implementation of SadTalker based on its [original github](https://github.com/OpenTalker/SadTalker).

## Introduction
SadTalker is a novel system for a stylized audio-driven single image talking face animation using the generated realistic 3D motion coefficients.

## Data and Preparation

Items to downloads:

> 1. checkpoints
> 2. examples (audio & image)

Data Structure:
<details>
  <summary>data tree</summary>
    
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
    examples/
    ├── driven_audio
    │   ├── bus_chinese.wav
    │   ├── chinese_news.wav
    │   ├── chinese_poem1.wav
    │   ├── chinese_poem2.wav
    │   ├── deyu.wav
    │   ├── eluosi.wav
    │   ├── fayu.wav
    │   ├── imagine.wav
    │   ├── itosinger1.wav
    │   ├── japanese.wav
    │   ├── RD_Radio31_000.wav
    │   ├── RD_Radio34_002.wav
    │   ├── RD_Radio36_000.wav
    │   └── RD_Radio40_000.wav
    ├── ref_video
    │   ├── WDA_AlexandriaOcasioCortez_000.mp4
    │   └── WDA_KatieHill_000.mp4
    └── source_image
        ├── art_0.png
        ├── art_10.png
        ├── art_11.png
        ├── art_12.png
        ├── art_13.png
        ├── art_14.png
        ├── art_15.png
        ├── art_16.png
        ├── art_17.png
        ├── art_18.png
        ├── art_19.png
        ├── art_1.png
        ├── art_20.png
        ├── art_2.png
        ├── art_3.png
        ├── art_4.png
        ├── art_5.png
        ├── art_6.png
        ├── art_7.png
        ├── art_8.png
        ├── art_9.png
        ├── full3.png
        ├── full4.jpeg
        ├── full_body_1.png
        ├── full_body_2.png
        ├── happy1.png
        ├── happy.png
        ├── people_0.png
        ├── sad1.png
        └── sad.png
    ```
</details>

## Inference
```bash
python inference.py --source_image examples/source_image/people_0.png --driven_audio examples/driven_audio/imagine.wav
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
