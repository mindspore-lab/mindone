# Repo Structure

```plaintext
opensora_cai
├── README.md
├── assets
│   └── texts                               -> prompts used for text-conditioned generation
│       └── t2v_samples.txt
├── configs                                 -> configs for training & inference
│   ├── opensora
│   │   ├── inference
│   │   │   ├── stdit_256x256x16.yaml
│   │   │   ├── stdit_512x512x16.yaml
│   │   │   └── stdit_512x512x64.yaml
│   │   └── train
│   │       ├── stdit_256x256x16.yaml
│   │       ├── stdit_256x256x16_ms.yaml    -> training receipe optimized for MindSpore
│   │       ├── stdit_512x512x16.yaml
│   │       ├── stdit_512x512x64.yaml
│   │       └── stdit_512x512x64_ms.yaml    -> training receipe optimized for MindSpore:
│   └── opensora-v1-1
├── docs
│   ├── config.md
│   ├── quick_start.md
│   └── structure.md
├── opensora
│   ├── datasets
│   │   ├── t2v_dataset.py
│   │   └── text_dataset.py
│   ├── models
│   │   ├── vae                             -> VAE as image encoder
│   │   ├── text_encoder                    -> Text encoder
│   │   ├── layers                          -> Common layers
│   │   └── stidit                          -> STDiT models
│   ├── pipelines
│   │   ├── __init__.py
│   │   ├── infer_pipeline.py               -> efficient inference pipeline for MindSpore
│   │   └── train_pipeline.py               -> efficient training pipeline for MindSpore
│   ├── schedulers
│   │   └── iddpm                           -> IDDPM for training and inference
│   │       ├── __init__.py
│   │       ├── diffusion_utils.py
│   │       ├── gaussian_diffusion.py
│   │       ├── respace.py
│   │       └── timestep_sampler.py
│   └── utils
├── requirements.txt
├── scripts
│   ├── args_train.py
│   ├── infer_t5.py
│   ├── infer_vae.py
│   ├── infer_vae_decode.py
│   ├── inference.py                        -> diffusion inference script
│   ├── run                                 -> scripts for quick running
│   └── train.py                            -> diffusion training script
├── tests                                   -> tests for the project
└── tools                                   -> tools for checkpoint conversion and visualization
```
