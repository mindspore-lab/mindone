# DiT Infer Acceleration

## Major Features
* Training-free
* Maximum 50% speed-up on transformer-based diffusion models
* 30% speedup with less than 1% accuracy loss using DiTCache and PromptGate
* Easy integration into DiT

## Performance
Stable Diffusion 3(SD3) uses the prompt "A cat holding a sign that says hello world" to run on Ascend 910B* at *graph mode*, and the performance data is as follows, which is spending time per image(s/img).

|Base model|DiTCache|DiTCache+PromptGate|DiTCache+PromptGate+ToDo|
|:--:|:--:|:--:|:--:|
|5.905|4.483|4.043|3.621|


## Introduce

>We examined the accelerating effects of multiple algorithms on DiT models.Taking Stable Diffusion 3 (SD3) as an example, We achieve a maximum of 🚀1.5x training-free acceleration. Combining DiTCache and PromptGate results in 1.3x acceleration with negligible precision loss, such that the FID increases from 21.62 to 21.97 and Clip-score remains unchanged. Adding ToDo to the package yields a total of 1.5 acceleration but the visual sense of the image may be slightly degraded.

## Algorithm Support

### DiTCache
>The denosing process of diffusion models are iterative. It has been observed that the feature values of certain hidden layers in the later denoising steps are similar to their precedents. Based on this observation, DiTCache is applied to the later denoising steps, which calculates and caches the differences in feature values between a starting block (e.g., Block1) and an ending block (e.g., Block10) in the first step of every two denoising steps, and then the cached difference is used to infer the feature values of the corresponding middle blocks (i.e., Block1~Block10) in the following step, thereby reduing a large amount of calculations. Applied this algorithm to SD3 results in an almost lossless 1.2x speedup.


### PromptGate
>This algorithm is inspired by [TGate(Temporally Gating)](https://github.com/HaozheLiu-ST/T-GATE)<sup>[1]</sup>. TGate found that, in the earlier stage of denosing, referred to as the "semantic-planning" phase, the prompt text has a more significant impact on the formation of image features; while in the later, so-called "fidelity-improving" phase, the attention calculation for the alignment of text and image is redundant. To reduce such redundancy, TGate ignores thr positive prompt in the fidelity-improving phase to reduce the amount of cross-attention calculation. We adapt TGate to accommodate the differences bewteen SD3 and the models TGate was developed on. Different from TGate, PromptGate removes the **negative prompt** instead of the positive prompt from the multi-modal self-attention calculation in the fidelity-improving phase. PomptGate combined with DiTCache achieves a total of 30% acceleration on SD3.

### ToDo
>ToDo(Token Downsample)<sup>[2]</sup> is an improved algorithm based on ToMe(Token Merge)<sup>[3]</sup>. While both algorithms aim to reduce computational load by reducing the length of input tokens in attention calculations, ToDo found it more efficient to directly downsample the K, V matrices before attention calculations. The original ToDo was applied to the cross-attention blocks in UNets, which is drastically different from the MMDiT structures in SD3. We adapt the original algorithm for MMDiT by downsampling the K, V matrices of the image features before they are concatenated with text features.


## Cited Work

[1] [Zhang, Wentian, Liu, Haozhe, Xie, Jinheng, Faccio, Francesco, Shou, Mike Zheng, and Schmidhuber, Jürgen. "Cross-Attention Makes Inference Cumbersome in Text-to-Image Diffusion Models", 2024.](https://arxiv.org/abs/2404.02747v1)

[2] [Smith, Ethan, Saxena, Nayan, and Saha, Aninda. "ToDo: Token Downsampling for Efficient Generation of High-Resolution Images", 2024.](https://arxiv.org/abs/2402.13573v3)

[3] [Bolya, Daniel, and Judy Hoffman. "Token Merging for Fast Stable Diffusion". *CVPR Workshop on Efficient Deep Learning for Computer Vision*, 2023.](https://arxiv.org/abs/2303.17604)
