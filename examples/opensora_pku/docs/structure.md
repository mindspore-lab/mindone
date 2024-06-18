# Repo Structure
```plaintext
opensora_pku/
├── README.md
├── docs
│   └── structure.md
├── examples
│   ├── prompt_list_0.txt                           -> prompts used for text-conditioned generation
│   ├── rec_imvi_vae.py                             -> causal vae video reconstruction
│   └── rec_video_vae.py                            -> causal vae video reconstruction given input video folder
├── opensora
│   ├── dataset
│   │   ├── image_dataset.py
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── t2v_dataset.py
│   │   ├── text_dataset.py
│   │   └── transform.py
│   ├── eval                                        -> Evaluate multiple metrics
│   ├── models
│   │   ├── ae
│   │   │   ├── imagebase                           -> image-based VAE
│   │   │   │   └── vae
│   │   │   │       └── vae.py
│   │   │   ├── __init__.py
│   │   │   └── videobase                           -> video-based VAE
│   │   │       ├── causal_vae
│   │   │       │   ├── __init__.py
│   │   │       │   └── modeling_causalvae.py       -> the model definition of causal vae model
│   │   │       ├── configuration_videobase.py
│   │   │       ├── dataset_videobase.py
│   │   │       ├── __init__.py
│   │   │       ├── losses
│   │   │       │   ├── discriminator.py
│   │   │       │   ├── __init__.py
│   │   │       │   ├── lpips.py
│   │   │       │   ├── net_with_loss.py
│   │   │       │   └── perceptual_loss.py
│   │   │       ├── modeling_videobase.py
│   │   │       ├── modules
│   │   │       └── trainer_videobase.py
│   │   ├── captioner
│   │   ├── diffusion
│   │   ├── frame_interpolation
│   │   ├── super_resolution
│   │   └── text_encoder
│   │       ├── clip.py
│   │       ├── __init__.py
│   │       ├── t5_encoder.py
│   │       └── t5.py
│   ├── sample
│   │   ├── pipeline_videogen.py
│   │   └── sample_t2v.py
│   ├── train
│   │   ├── commons.py
│   │   ├── train_causalvae.py
│   │   └── train_t2v.py
│   └── utils
├── requirements.txt
├── scripts
│   ├── causalvae
│   │   ├── eval.sh
│   │   ├── gen_video.sh                            -> Reconstruct videos given an input video folder
│   │   ├── reconstruction.sh                       -> Reconstruct an input video
│   │   ├── release.json
│   │   └── train.sh                                -> Train the causal vae model
│   ├── model_conversion
│   │   └── convert_all.sh
│   └── text_condition
│       ├── sample_image.sh
│       ├── sample_video.sh                         -> Run text-to-video inference
│       ├── train_videoae_17x256x256.sh
│       ├── train_videoae_65x256x256.sh
│       └── train_videoae_65x512x512.sh
└── tools
    └── model_conversion
```
