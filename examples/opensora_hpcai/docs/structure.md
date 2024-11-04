# Repo Structure

```plaintext
opensora_hpcai
├── README.md
├── assets
│   └── texts                               -> prompts used for text-conditioned generation
│       └── t2v_samples.txt
├── configs                                 -> configs for training & inference
│   ├── opensora
│   ├── opensora-v1-1
│   │   ├── inference
│   │   └── train
│   │── opensora-v1-2
│   │   ├── inference
│   │   └── train
│   └── cogvideox 
│       └── train
│          └── cogvideo5b_720x480x48.yaml    -> **high-quality video finetuning config; zero-2, context parallel ** 
├── opensora
│   ├── datasets
│   │   ├── t2v_dataset.py		    -> **frame packing (optional)**
│   │   └── text_dataset.py
│   ├── models
│   │   ├── vae                             -> ** causal vae 3d, context parallel or vae tiling/caching **
│   │   ├── text_encoder                   
│   │   ├── layers                          
│   │   ├── stidit                          -> STDiT models
│   │   └── cogvideox                       -> ** cogvideox dit arch (key changes: full attn, RoPE-3d)** 
│   ├── pipelines
│   │   ├── __init__.py
│   │   ├── infer_pipeline.py               -> efficient inference pipeline for MindSpore
│   │   └── train_pipeline.py               -> ** add v-pred, zero-SNR **
│   ├── schedulers
│   │   ├── rectified_flow.py
│   │   └── iddpm                           -> IDDPM for training and inference
│   │       ├── __init__.py
│   │       ├── diffusion_utils.py
│   │       ├── gaussian_diffusion.py
│   │       ├── respace.py
│   │       └── timestep_sampler.py	    -> ** add Explicit Uniform Sampling ** 
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
