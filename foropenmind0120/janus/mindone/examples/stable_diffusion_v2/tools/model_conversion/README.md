# Stable Diffusion Weights Conversion

## Purpose

The purpose of this tool is to cater for the needs:

1. source-agnostic fine-tuning

- pre-train on MindSpore + Ascend, fine-tune on PyTorch + GPU, and vice versa

2. infra-agnostic inference

- train on MindSpore + Ascend, infer on PyTorch + GPU, and vice versa

### Headsup

Always backup your original weight files before any conversion.

### Preperation

```shell
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/examples/stable_diffusion_v2
```

### Example usage:

- [x] convert PyTorch SD v2.* weights to MindSpore

```shell
python tools/model_conversion/convert_weights.py \
--source PATH_TO_TORCH.pt \
--target PATH_TO_SAVE_MS.ckpt \
--model sdv2 \
--source_version pt
```

- [x] convert MindSpore SD v2.* weights to PyTorch (diffusers) (will overwrite old bin files)

```shell
python tools/model_conversion/convert_weights.py \
--source PATH_TO_MS.ckpt \
--target FOLDER_TO_SAVE_TORCH \
--model diffusersv2 \
--source_version ms
```

- [x] convert MindSpore SD v2.* weights to PyTorch (full ckpt)

```shell
python tools/model_conversion/convert_weights.py \
--source PATH_TO_MS.ckpt \
--target PATH_TO_SAVE_TORCH.pt \
--model sdv2 \
--source_version ms
```

- [x] convert PyTorch SD v1.* weights to MindSpore

```shell
python tools/model_conversion/convert_weights.py \
--source PATH_TO_TORCH.pt \
--target PATH_TO_SAVE_MS.ckpt \
--model sdv1 \
--source_version pt
```

- [] convert MindSpore SD v1.* weights to PyTorch

- [] convert PyTorch ControNet v1.* weights to MindSpore
