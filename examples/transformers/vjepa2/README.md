<!-- # https://github.com/huggingface/transformers/pull/38788 -->


# V-JEPA 2

V-JEPA 2 is a self-supervised approach to training video encoders developed by FAIR, Meta. Using internet-scale video data, V-JEPA 2 attains state-of-the-art performance on motion understanding and human action anticipation tasks. V-JEPA 2-AC is a latent action-conditioned world model post-trained from V-JEPA 2 (using a small amount of robot trajectory interaction data) that solves robot manipulation tasks without environment-specific data collection or task-specific training or calibration.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vjepa.gif" alt="drawing" width="600"/>
</div>

You can find all original V-JEPA2 checkpoints under the [V-JEPA 2](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6) collection.


# Get Started
## Requirements
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.5.0 | 24.1.RC3 | 7.5.T11.0 | 8.0.0.beta1|
|2.6.0 | 24.1.RC3 | 7.5.T11.0 | 8.0.0.beta1|

### Installation
```
# with installed mindone
cd examples/transformers/vjepa2
bash install.sh
```

## Usage example

The script `extract_feat.py` shows how to load the V-JEPA 2 model for feature extraction.

```bash
python extract_feat.py
```

V-JEPA 2 can also be finetuned for video classification. In the following snippet, `classify.py` shows how use finetuned on Something-Something-V2 video classification model.

```bash
python classify.py
```

## Inference Performance

Experiments are tested on ascend 910* with pynative mode.

- mindspore 2.5.0

|model| precision |  resolution| fa | s/step | weight
|---|---|---|---|---|---|
|vjepa2-vitl-fpc64-256| fp32 | 64x3x256x256 | OFF | 2.16  | [weight](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) |
|vjepa2-vitl-fpc64-256| bf16 | 64x3x256x256 | ON | 0.71  | [weight](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) |
|vjepa2-vitl-fpc64-256| fp16 | 64x3x256x256 | ON | 0.63  | [weight](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) |


- mindspore 2.6.0

|model| precision |  resolution| fa | s/step | weight
|---|---|---|---|---|---|
|vjepa2-vitl-fpc64-256| fp32 | 64x3x256x256 | OFF | 2.19  | [weight](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) |
|vjepa2-vitl-fpc64-256| bf16 | 64x3x256x256 | ON | 0.74  | [weight](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) |
|vjepa2-vitl-fpc64-256| fp16 | 64x3x256x256 | ON | 0.63  | [weight](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) |
