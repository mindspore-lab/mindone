# Stable Diffusion Weights Conversion

## Train anywhere, run anywhere

Example usage:

```shell
# preperation
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/examples/stable_diffusion_v2
```

- [x] convert torch SD v1.* weights to mindspore

```shell
python tools/convert_weights.py \
--source PATH_TO_TORCH.pt
--target PATH_TO_SAVE_MS.ckpt
--model sdv1
--source_version pt
```

- [x] convert torch SD v2.* weights to mindspore

```shell
python tools/convert_weights.py \
--source PATH_TO_TORCH.pt
--target PATH_TO_SAVE_MS.ckpt
--model sdv2
--source_version pt
```

- [] convert mindspore SD v1.* weights to torch

- [x] convert mindspore SD v2.* weights to torch

```shell
python tools/convert_weights.py \
--source PATH_TO_MS.ckpt
--target PATH_TO_SAVE_TORCH.pt
--model sdv2
--source_version ms
