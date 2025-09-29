## Evaluating MindONE Models

MindONE models focuses on the area of content generation, which could be hard to choose one over the other simply based on qualitative evaluations. In this document, we provide several quantitative methos to evaluate different generating tasks.

## Quantitative Evaluation

In this section, we will walk you through how to evaluate image generation using:

- Inception Score
- FID
- CLIP Score
- CLIP directional similarity

### Supported metrics table

|                  | Support | Functional |
| ---------------- |---------| ---------- |
| **IS**           | √       | √          |
| **FID**          | √       | √          |
| **CLIP Score**   | √       | √          |
| **CLIP dir sim** | √       | √          |
MindONE metrics are developed and tested under mindspore 2.2, raise an issue if you prefer r2.3.

### Text-guided image generation

[CLIP Score](https://arxiv.org/abs/2104.08718) is a reference free metric that can be used to evaluate the correlation between a generated caption for an image and the actual content of the image. It has been found to be highly correlated with human judgement. The metric is defined as:
$$
\text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)
$$
which corresponds to the cosine similarity between visual CLIP embedding $E_i$ for an image $i$ and textual CLIP embedding $E_C$ for an caption $C$. The score is bound between 0 and 100 and the closer to 100 the better.

The CLIP Processor and CLIP Model is generated with [mindformers](https://github.com/mindspore-lab/mindformers), where as pretrained weights are provided. We are working on better performance with [mindnlp](https://github.com/mindspore-lab/mindnlp) developers, whose weights are identical with transformers, to produce closer results with torchmetrics.

#### CLIP Score functional using cases

```python
import mindspore as ms  
from mindone.metrics.functional import clip_score
images = ms.ops.randint(0, 255, (2, 3, 244, 244)).to(ms.uint8)  
text = ["a photo of a cat", "a photo of a dog"]
score = clip_score(images, text)
print(f"CLIP score: {score}")
# CLIP score: [19.463135]
```

You can calculate clip score with a 4D tensor and a list of string, notice that the shape of the first dimension of the image tensor must equals to the length of text list.

You could use update and compute for multi-batches calculation:

```Python
import mindspore as ms
from mindone.metrics.multimodal.clip_score import ClipScore

image1 = ms.ops.randint(0, 255, (3, 244, 244), seed=123).to(ms.uint8)
image2 = ms.ops.randint(0, 255, (3, 244, 244), seed=123).to(ms.uint8)
text = "a photo of a cat"
metric = ClipScore()
metric.update(image1, text)
metric.update(image2, text)
score = metric.compute()
print(f"CLIP score: {score}")
# CLIP score: [20.441158]
```

You could also use different pretrained clip model:
```python
score = clip_score(images, text, model_name_or_path="clip_vit_b_16")
```

### Image-conditioned text-to-image generation

CLIP Directional Similarity(first mentioned in [StyleGAN-NADA](https://arxiv.org/abs/2108.00946)) metric is to measure the consistency of the change between the two images (in CLIP space) with the change between the two image captions. The metric is defined as:
$$
\text{CLIPDirectionalSimilarity}(I_1, I_2, C_1, C_2) = cos(E_{I_1} - E_{I_2}, E_{C_1} - E_{C_2})
$$
which corresponds to the cosine similarity between the difference of visual `CLIP`_ embeddings of two images and textual CLIP embeddings of two texts. The higher the CLIP directional similarity, the better it is.

Samewise, the CLIP Processor and CLIP Model is generated with [mindformers](https://github.com/mindspore-lab/mindformers), where as pretrained weights are provided. We are working on better performance with [mindnlp](https://github.com/mindspore-lab/mindnlp) developers, whose weights are identical with transformers, to produce closer results with torchmetrics.

#### CLIP Directional Similarity functional using cases

```python
import mindspore as ms  
from mindone.metrics.functional import clip_directional_similarity  
original_image_ms = ms.ops.randint(0, 255, (3, 224, 224)).to(ms.uint8)  
generated_image_ms = ms.ops.randint(0, 255, (3, 224, 224)).to(ms.uint8)  
original_caption_ms = "a photo of cat"  
modified_caption_ms = "a photo of dog"  
output = clip_directional_similarity(original_image_ms, generated_image_ms, original_caption_ms, modified_caption_ms)  
print(f"CLIP directional similarity: {output}")
# CLIP directional similarity: -0.030828297
```

You can calculate clip directional with 2 4D tensors and 2 lists of string, notice that the shape of the first dimension of the image tensors must equals to the length of text lists.

You could use update and compute for multi-batches calculation:

```Python
import mindspore as ms
import numpy as np
from mindone.metrics.multimodal.clip_directional_similarity import ClipDirectionalSimilarity

np.random.seed(123)
origin_image_np = np.random.randint(0, 255, (3, 224, 224))
generated_image_np = np.random.randint(0, 255, (3, 224, 224))
origin_text = "a photo of cat"
edited_text = "a photo of dog"
original_image_ms = ms.Tensor(origin_image_np).to(ms.uint8)
original_caption_ms = origin_text
edited_image_ms = ms.Tensor(generated_image_np).to(ms.uint8)
modified_caption_ms = edited_text
metric = ClipDirectionalSimilarity()
metric.update(original_image_ms, edited_image_ms, original_caption_ms, modified_caption_ms)
output = metric.compute()
print(f"CLIP directional similarity: {output}")
# CLIP directional similarity: -0.028446209
```

You could also use different pretrained clip model:
```python
output = clip_directional_similarity(i1, i2, t1, t2, model_name_or_path="clip_vit_b_16")
```

### Class-conditioned image generation

Class-conditioned generative models are usually pre-trained on a class-labeled dataset such as ImageNet-1k. Popular metrics for evaluating these models include Fréchet Inception Distance (FID) and Inception Score (IS).

#### FID

Calculate Fréchet inception distance (FID_) which is used to access the quality of generated images.
$$
FID = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})
$$
where $\mathcal{N}(\mu, \Sigma)$ is the multivariate normal distribution estimated from [Inception v3](https://arxiv.org/abs/1512.00567) features calculated on real life images and $\mathcal{N}(\mu_w, \Sigma_w)$ is the multivariate normal distribution estimated from Inception v3 features calculated on generated (fake) images. The metric was originally proposed in [# Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).
#### FID using cases

```python
import mindspore as ms  
from mindone.metrics.functional import fid  
imgs_dist1 = ms.ops.randint(0, 200, (100, 3, 299, 299)).to(ms.uint8)  
imgs_dist2 = ms.ops.randint(100, 255, (100, 3, 299, 299)).to(ms.uint8)  
output = fid(imgs_dist1, imgs_dist2, feature=64)
print(f"fid: {output}")
# fid: 12.646194685028123
```

You can calculate fid with two 4D tensors, notice that the shape of the first dimension of the image tensors must be identical.

You could use update and compute for multi-batches calculation:

```Python
import mindspore as ms
from mindone.metrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(feature=64)
imgs_dist1 = ms.ops.randint(0, 200, (100, 3, 299, 299), seed=123).to(ms.uint8)
imgs_dist2 = ms.ops.randint(100, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)
fid.update(imgs_dist1, real=True)
fid.update(imgs_dist2, real=False)
output = fid.compute()
print(f"fid: {output}")
# fid: 12.627565920736048
```

You could also use different shape feature extractors (64, 192, 768, 2048):
```python
output = fid(imgs_dist1, imgs_dist2, feature=2048)
```

#### Inception Score

Calculate the Inception Score (IS) which is used to access how realistic generated images are.

$$IS = exp(\mathbb{E}_x KL(p(y | x ) || p(y)))
$$

where $KL(p(y | x) || p(y))$ is the KL divergence between the conditional distribution $p(y|x)$  
and the marginal distribution $p(y)$. Both the conditional and marginal distribution is calculated from features extracted from the images. The score is calculated on random splits of the images such that both a mean and standard deviation of the score are returned. The metric was originally proposed in [Imporved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
#### IS functional using cases

```python
import mindspore as ms  
from mindone.metrics.functional import inception_score  
imgs = ms.ops.randint(0, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)  
output = inception_score(imgs)  
print(f"IS: {output}")
# IS: (Tensor(shape=[], dtype=Float32, value= 1.0607), Tensor(shape=[], dtype=Float32, value= 0.0186892))
```

You could use update and compute for multi-batches calculation:

```Python
import mindspore as ms
from mindone.metrics.image.inception_score import InceptionScore

# splits equals to 1 to avoid shuffle operation generating different results
inception_score = InceptionScore(splits=1)
imgs1 = ms.ops.randint(0, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)
imgs2 = ms.ops.randint(0, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)
inception_score.update(imgs1)
inception_score.update(imgs2)
output = inception_score.compute()
print(f"IS: {output}")
# IS: (Tensor(shape=[], dtype=Float32, value= 1.0694), Tensor(shape=[], dtype=Float32, value= 0))
```
