<h1 align="center">FBCache_ms: Forward Block Caching for Accelerated Diffusion Model Inference</h1>

# Overview

This repository includes the MindSpore pipeline of Flux.1 that is accelerated by First-Block Cache (FBCache) and Taylorseer, refactored to be compatible with the MindSpore Graph Mode.

Caching significantly speeds up the inference process of diffusion models by reusing intermediate computation results across timesteps. Most caching algorithms like DeepCache and Delta Cache require parameter tuning based on experimental data analysis and prior knowledge, which can be time-consuming and lack generalizability. FBCache circumvents this problem by using the residual output of the first transformer block as the cache indicator, which allows dynamic caching.

Feature caching inevitably introduces errors in the denoising process, thereby harming the generation quality. Taylorseer aims to address this problem by predicting features at future timesteps based on the cached features instead of simply reusing them. Specifically, Taylorseer employs a differential method to approximate the higher-order derivatives of features and predict features at future time steps with Taylor series expansion.

The implementation of these two algorithms, combined with the Graph Compilation Mode of MindSpore, significantly speeds up the inference process of Flux.1 while maintaining high generation quality.

## 📦 Requirements

<div align="center">

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.7.0   |   24.1.RC3    | 7.6.0.1.220 |    8.0.RC3.beta1    |

</div>

1. Install
   [CANN 8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) and MindSpore according to the [official instructions](https://www.mindspore.cn/install).
2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```
3. Install mindone
    ```shell
    cd mindone
    pip install -e .
    ```
   Try `python -c "import mindone"`. If no error occurs, the installation is successful.

## 🚀 Quick Start

### Step 1: Prepare Flux Model Checkpoint

First, download the Flux model checkpoint from [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-schnell) or [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

### Step 2: Run Inference with FBCache

Navigate to the FBCache_ms directory and run the evaluation script to generate images with FBCache acceleration:

```bash
cd examples/accelerated_dit_pipelines/FBCache_ms/eval
python run_flux_withFBCache_pipeline.py \
  --ckpt /path/to/FLUX.1-dev \
  --prompt "A cat holding a sign that says hello world" \
  --image_size 1024 1024 \
  --save_path ./results \
  --taylorseer_derivative 2 \
  --residual_diff_threshold 0.08 \
  --use_graph_mode
```

### Key Parameters

- `--ckpt`: Path to the Flux model checkpoint
- `--prompt`: Text prompt for image generation
- `--image_size`: Output image size (height, width)
- `--taylorseer_derivative`: Order of Taylor series approximation for cache (0 = disabled)
- `--residual_diff_threshold`: Threshold for determining cache reuse eligibility, higher means more aggressive caching
- `--use_graph_mode`: Enable MindSpore graph mode for additional acceleration

## 📊 Performance Benchmarks

Experiments are tested on Ascend Atlas 800T A2 machines.

### Inference Speed Comparison

| Configuration | Image Size | Steps | Time per Step (s) | Total Time (s) | Speedup |
|---------------|------------|-------|-------------------|----------------|--------|
| Baseline (PyNative Mode)      | 1024x1024  | 28    | 0.63              | 17.5           | 1.0x   |
| Baseline (Graph Mode) | 1024x1024  | 28    | 0.45              | 12.7           | 1.38x  |
| FBCache(0.08) + Graph Mode | 1024x1024  | 28    | 0.27              | 7.68           | 2.28x  |
| FBCache(0.12) + Graph Mode | 1024x1024  | 28    | 0.23              | 6.35           | 2.76x  |

### Generation Quality Comparison
FID score based on MSCOCO 2017 Validation dataset (5000 images)
| Configuration | Image Size | Steps | Total Time (s) | FID |
|---------------|------------|-------|-------------------|----------------|--------|
| Baseline| 1024x1024  | 28    |  12.7           | 34.46  |
| FBCache（0.08) + Taylorseer | 1024x1024  | 28    |  7.68           | 34.48  |
| FBCache（0.12) + Taylorseer| 1024x1024  | 28    |6.35           | 34.73  |

## 💡 How FBCache Works

1. **Residual Tracking**: The framework tracks the residual differences between consecutive timesteps.
2. **Cache Eligibility**: When the residual difference falls below a configurable threshold, subsequent transformer blocks are skipped.
3. **Cache Reuse**: Pre-computed results from previous timesteps are reused, significantly reducing computation.
4. **Taylor Series Approximation**: For further acceleration, an optional Taylor series approximation can be applied to predict residuals between timesteps.

### 🛠️ Project Structure

```
FBCache_ms/
├── eval/                     # Evaluation scripts
│   └── run_flux_withFBCache_pipeline.py  # Main evaluation script
├── models/                   # Model definitions
│   ├── __init__.py
│   └── transformer_flux_withFBCache.py  # FBCache-enabled transformer for Flux.1
├── pipeline_flux.py          # Modified Flux pipeline with FBCache integration
├── tests/                    # Unit tests
│   ├── flux_accuracy_test.py
│   ├── generate_clip_score_datas.py
│   └── test_accuracy_data.py
└── utils.py                  # Utility functions and CacheContext implementation
```
### Core Components

#### 1. CacheContext

Manages the caching mechanism and Taylor series approximation:

```python
from examples.accelerated_dit_pipelines.FBCache_ms.utils import CacheContext

# Initialize cache context
cache_context = CacheContext(
    batch_size=batch_size,
    seq_len=sequence_length,
    inner_dim=hidden_dimension,
    taylorseer_derivative=1  # Enable 1st order Taylor approximation
)
```

#### 2. FBCache_transformer_construct

Enhanced transformer forward pass with caching logic:

```python
from examples.accelerated_dit_pipelines.FBCache_ms.models.transformer_flux_withFBCache import FBCache_transformer_construct

# Register the enhanced forward method to the transformer
model.transformer.construct = MethodType(FBCache_transformer_construct, model.transformer)
model.transformer.cache_context = cache_context
```
### 📝 Notes

1. **Quality-Speed Tradeoff**: Increasing the `residual_diff_threshold` can improve speed but may impact image quality.
2. **Memory Considerations**: FBCache increases memory usage slightly to store cached intermediate results. Increasing the order of Taylor series approximation may further increase memory consumption.
3. **Prompt Sensitivity**: Results may vary depending on the complexity of the prompt and the content being generated.

## Acknowledgement

The FBCache implementation is adapted from https://github.com/chengzeyi/ParaAttention/tree/main/src/para_attn/first_block_cache.
The Taylorseer implementation is based on [From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/html/2503.06923v1).
