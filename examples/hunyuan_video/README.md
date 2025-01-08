# Hunyuan Video


## Quick Start

### Installation

### Checkpoints

Please download all checkpoints and convert them into MindSpore checkpoints following this [instruction](./ckpts/README.md).

### Run VAE reconstruction

To run a video reconstruction using the CausalVAE, please use the following command:
```bash
python hyvideo/rec_video.py \
  --video_path input_video.mp4 \
  --rec_path rec.mp4 \
  --height 360 \
  --width 640 \
  --num_frames 33 \
```
The reconstructed video is saved under `./samples/`.


### Run Text-to-Video Inference




### Run Image-to-Video Inference


## Train


## Evaluation


### VAE Evaluation

To evaluate VAE's PNSR, please download MCL_JCV dataset from this [URL](https://mcl.usc.edu/mcl-jcv-dataset/), and place the videos under `datasets/MCL_JCV`.

Now, to run video reconstruction on a video folder, please run:

```bash
python hyvideo/rec_video_folder.py \
  --real_video_dir datasets/MCL_JCV \
  --generated_video_dir datasets/MCL_JCV_generated \
  --height 360 \
  --width 640 \
  --num_frames 33 \
```

Afterwards, you can evaluate the PSNR via:
```bash
bash hyvideo/eval/scripts/cal_psnr.sh
```
