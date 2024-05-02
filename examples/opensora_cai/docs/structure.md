# Repo Structure

```plaintext
opensora_cai
├── README.md
├── assets
│   └── texts
│       └── t2v_samples.txt
├── configs
│   ├── opensora
│   │   ├── inference
│   │   │   ├── stdit_256x256x16.yaml
│   │   │   ├── stdit_512x512x16.yaml
│   │   │   └── stdit_512x512x64.yaml
│   │   └── train
│   │       ├── stdit_256x256x16.yaml
│   │       ├── stdit_256x256x16_ms.yaml
│   │       ├── stdit_512x512x16.yaml
│   │       ├── stdit_512x512x64.yaml
│   │       └── stdit_512x512x64_ms.yaml
│   └── opensora-v1-1
├── docs
│   ├── config.md
│   └── structure.md
├── opensora
│   ├── datasets
│   │   ├── t2v_dataset.py
│   │   └── text_dataset.py
│   ├── models
│   │   ├── autoencoder.py
│   │   ├── flan_t5_large
│   │   │   ├── mininlp
│   │   │   │   ├── activations.py
│   │   │   │   ├── configs.py
│   │   │   │   ├── download.py
│   │   │   │   ├── legacy.py
│   │   │   │   ├── mixins.py
│   │   │   │   ├── pretrained_config.py
│   │   │   │   ├── pretrained_model.py
│   │   │   │   └── pretrained_tokenizer.py
│   │   │   ├── t5.py
│   │   │   └── t5_config.py
│   │   ├── layers
│   │   │   └── blocks.py
│   │   ├── stdit.py
│   │   ├── t5.py
│   │   └── text_encoders.py
│   ├── pipelines
│   │   ├── __init__.py
│   │   ├── infer_pipeline.py
│   │   └── train_pipeline.py
│   ├── schedulers
│   │   └── iddpm
│   │       ├── __init__.py
│   │       ├── diffusion_utils.py
│   │       ├── gaussian_diffusion.py
│   │       ├── respace.py
│   │       └── timestep_sampler.py
│   └── utils
│       ├── cond_data.py
│       ├── load_models.py
│       ├── model_utils.py
│       └── util.py
├── quick_start.md
├── requirements.txt
├── scripts
│   ├── args_train.py
│   ├── infer_t5.py
│   ├── infer_vae.py
│   ├── infer_vae_decode.py
│   ├── inference.py
│   ├── run
│   │   └── run_train_512x512x64_parallel_ge.sh
│   └── train.py
├── tests
│   ├── test_data.py
│   ├── test_stdit.py
│   ├── test_stdit_block.py
│   └── test_stdit_ms.py
└── tools
    ├── ckpt_combine.py
    ├── convert_pt2ms.py
    ├── plot.py
    ├── t5_convert.py
    └── vae_converter.py
```
