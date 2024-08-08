# Fatezero

## 代码根目录

```
cd mindone/examples/stable_diffusion_v2
```



## Installation & Preparation

### Environment and Dependency

**Device:** Ascend 910

**Framework:** MindSpore >= 2.1

Install dependent packages by:

```
pip install -r requirements.txt
```

### Pretrained Checkpoint

*如果使用新视频，需要使用tuneavideo做fine-tuning，并将结果作为pretrained checkpoint*

- SD2.0 Download [sd-500.ckpt](../../../sd-500.ckpt) and put it under ./ folder

  

### Dataset Preparation  

The video for base should follow the file structure below

```
./videos
```

## Run

```
python inference_fatezero.py --ms_mode 1 --config configs/v2-interface-fatezero-model.yaml --version 2.0 --video_path videos/jeep.mp4 --ckpt_path sd-500.ckpt --num_frames 8 --output_path output/ --source_prompt "a silver jeep driving down a curvy road in the countryside" --target_prompt "a Porsche car driving down a curvy road in the countryside" --ddim --sampling_steps 50
```



