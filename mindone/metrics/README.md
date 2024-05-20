## Evaluating MindONE Models

MindONE models focuses on the area of content generation, which could be hard to choose one over the other simply based on qualitative evaluations. In this document, we provide several quantitative methos to evaluate different generating tasks.

## Quantitative Evaluation

In this section, we will walk you through how to evaluate image generation using:

- Inception Score
- FID
- CLIP Score
- CLIP directional similarity

### Supported metrics table

|                  | Functional | Distributed (Ascend) | Graph Mode |
| ---------------- | ---------- | -------------------- | ---------- |
| **IS**           | √          | √                    | √          |
| **FID**          | √          | √                    | √          |
| **CLIP Score**   | √          | √                    | x          |
| **CLIP dir sim** | √          | √                    | x          |
MindONE metrics are developed and tested under mindspore 2.2, raise an issue if you prefer r2.3.

### Text-guided image generation

[CLIP Score](https://arxiv.org/abs/2104.08718) is a reference free metric that can be used to evaluate the correlation between a generated caption for an image and the actual content of the image. It has been found to be highly correlated with human judgement. The metric is defined as:
$$
\text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)
$$
which corresponds to the cosine similarity between visual CLIP embedding $E_i$ for an image $i$ and textual CLIP embedding $E_C$ for an caption $C$. The score is bound between 0 and 100 and the closer to 100 the better.

The CLIP Processor and CLIP Model is generated with [mindformers](https://github.com/mindspore-lab/mindformers), where as pretrained weights are provided. We are working on better performance with [mindnlp](https://github.com/mindspore-lab/mindnlp) developers, whose weights are identical with transformers, to produce closer results with torchmetrics.

#### CLIP Score functional using case

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

You could also use different pretrained clip model:
```python
score = clip_score(images, text, model_name_or_path="clip_vit_b_16")
```

#### CLIP Score distributed using case

CLIP Score metric supports distributed environment for large dataset cases. Achieved with mindspore framework, you need to build an available mindspore distributed environment first. We currently only support rank table startup with ascend devices. Raise an issue if you prefer other ways or devices.

Firstly, generate a rank table json file to specify the device you prefer to use, guided by [mindspore official docs](https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/rank_table.html). You can generate rank table automatically with [hccl_tools](https://gitee.com/mindspore/models/blob/r1.7/utils/hccl_tools/hccl_tools.py).

Then write a python script for metric calculating:

```python
import mindspore as ms
from mindone.metrics.clip_score import ClipScore
# set contexts, GRAPH_MODE is not supported
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
# init distributed environment for all gather operations
init()
# you could load in images and texts instead of generating random ones
imgs = ms.ops.randint(0, 255, (100, 3, 224, 224)).to(ms.uint8)
text = ["an image of dog" + str(idx) for idx in range(100)]
metric = ClipScore()
output = metric(imgs, text, model_name_or_path="clip_vit_l_14")
print("CLIP Score: ", output)
```

At last, write a shell script to startup calculating (script below is a 4 device case), the python script above is named 'distributed_text.py'. Notice that you may prefer copying checkpoint_download folder to avoid downloading multi times.

```shell
#!/bin/bash

export DEVICE_NUM=4
export RANK_SIZE=4
export RANK_TABLE_FILE="/home/conf/hccl_4p_4567_127.0.0.1.json"

pids=()
for((i=4;i<8;i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./distributed_test.py ./device$i
    cp -R ./checkpoint_download ./device$i/checkpoint_download
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$(expr $i - 4)
    echo "start training for device $DEVICE_ID, rank id: $RANK_ID"
    env > env$i.log
    python ./distributed_test.py > test$i.log 2>&1 &
    cd ../
done
```

### Image-conditioned text-to-image generation

CLIP Directional Similarity(first mentioned in [StyleGAN-NADA](https://arxiv.org/abs/2108.00946)) metric is to measure the consistency of the change between the two images (in CLIP space) with the change between the two image captions. The metric is defined as:
$$
\text{CLIPDirectionalSimilarity}(I_1, I_2, C_1, C_2) = cos(E_{I_1} - E_{I_2}, E_{C_1} - E_{C_2})
$$
which corresponds to the cosine similarity between the difference of visual `CLIP`_ embeddings of two images and textual CLIP embeddings of two texts. The higher the CLIP directional similarity, the better it is.

Samewise, the CLIP Processor and CLIP Model is generated with [mindformers](https://github.com/mindspore-lab/mindformers), where as pretrained weights are provided. We are working on better performance with [mindnlp](https://github.com/mindspore-lab/mindnlp) developers, whose weights are identical with transformers, to produce closer results with torchmetrics.

#### CLIP Directional Similarity functional using case

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

You could also use different pretrained clip model:
```python
output = clip_directional_similarity(i1, i2, t1, t2, model_name_or_path="clip_vit_b_16")
```

#### CLIP directional similarity distributed using case

CLIP directional similarity metric supports distributed environment for large dataset cases. Achieved with mindspore framework, you need to build an available mindspore distributed environment first. We currently only support rank table startup with ascend devices. Raise an issue if you prefer other ways or devices.

Firstly, generate a rank table json file to specify the device you prefer to use, guided by [mindspore official docs](https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/rank_table.html). You can generate rank table automatically with [hccl_tools](https://gitee.com/mindspore/models/blob/r1.7/utils/hccl_tools/hccl_tools.py).

Then write a python script for metric calculating:

```python
import mindspore as ms
from mindone.metrics.clip_directional_similarity import ClipDirectionalSimilarity
# set contexts, GRAPH_MODE is not supported
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
# init distributed environment for all gather operations
init()
# you could load in images and texts instead of generating random ones
imgs_A = ms.ops.randint(0, 255, (2, 3, 224, 224)).to(ms.uint8)
imgs_B = ms.ops.randint(0, 255, (2, 3, 224, 224)).to(ms.uint8)
text_A = ["an image of dog", "a cloudy day"]
text_B = ["an image of cat", "a sunny day"]
metric = ClipDirectionalSimilarity()
output = metric(imgs_A, imgs_B, text_A, text_B)
print("CLIP dir sim: ", output)
```

At last, write a shell script to startup calculating (script below is a 4 device case), the python script above is named 'distributed_text.py'. Notice that you may prefer copying checkpoint_download folder to avoid downloading multi times.

```shell
#!/bin/bash

export DEVICE_NUM=4
export RANK_SIZE=4
export RANK_TABLE_FILE="/home/conf/hccl_4p_4567_127.0.0.1.json"

pids=()
for((i=4;i<8;i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./distributed_test.py ./device$i
    cp -R ./checkpoint_download ./device$i/checkpoint_download
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$(expr $i - 4)
    echo "start training for device $DEVICE_ID, rank id: $RANK_ID"
    env > env$i.log
    python ./distributed_test.py > test$i.log 2>&1 &
    cd ../
done
```

### Class-conditioned image generation

Class-conditioned generative models are usually pre-trained on a class-labeled dataset such as ImageNet-1k. Popular metrics for evaluating these models include Fréchet Inception Distance (FID) and Inception Score (IS).

#### FID

Calculate Fréchet inception distance (FID_) which is used to access the quality of generated images.
$$
FID = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})
$$
where $\mathcal{N}(\mu, \Sigma)$ is the multivariate normal distribution estimated from [Inception v3](https://arxiv.org/abs/1512.00567) features calculated on real life images and $\mathcal{N}(\mu_w, \Sigma_w)$ is the multivariate normal distribution estimated from Inception v3 features calculated on generated (fake) images. The metric was originally proposed in [# Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).
#### FID functional using case

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

You could also use different shape feature extractors (64, 192, 768, 2048):
```python
output = fid(imgs_dist1, imgs_dist2, feature=2048)
```

#### FID distributed using case

FID metric supports distributed environment for large dataset cases. Achieved with mindspore framework, you need to build an available mindspore distributed environment first. We currently only support rank table startup with ascend devices. Raise an issue if you prefer other ways or devices.

Firstly, generate a rank table json file to specify the device you prefer to use, guided by [mindspore official docs](https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/rank_table.html). You can generate rank table automatically with [hccl_tools](https://gitee.com/mindspore/models/blob/r1.7/utils/hccl_tools/hccl_tools.py).

Then write a python script for metric calculating, notice that since ops.eigvals is not supported by ascend device under large shapes, so this part of calculation is separated from the construct as below:

```python
import mindspore as ms
from mindone.metrics.clip_directional_similarity import ClipDirectionalSimilarity
# set contexts, GRAPH_MODE is not supported
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
# init distributed environment for all gather operations
init()
# you could load in images and texts instead of generating random ones
real_imgs = ms.ops.randint(0, 200, (100, 3, 299, 299)).to(ms.uint8)
fake_imgs = ms.ops.randint(100, 255, (500, 3, 299, 299)).to(ms.uint8)
metric = FrechetInceptionDistance(feature=2048)
a, b = metric(real_imgs, fake_imgs)
output = metric.final_compute(a, b)
print("fid 2048: ", output)
```

At last, write a shell script to startup calculating (script below is a 4 device case), the python script above is named 'distributed_text.py'. Notice that you may prefer copying checkpoint_download folder to avoid downloading multi times.

```shell
#!/bin/bash

export DEVICE_NUM=4
export RANK_SIZE=4
export RANK_TABLE_FILE="/home/conf/hccl_4p_4567_127.0.0.1.json"

pids=()
for((i=4;i<8;i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./distributed_test.py ./device$i
    cp -R ./checkpoint_download ./device$i/checkpoint_download
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$(expr $i - 4)
    echo "start training for device $DEVICE_ID, rank id: $RANK_ID"
    env > env$i.log
    python ./distributed_test.py > test$i.log 2>&1 &
    cd ../
done
```

#### Inception Score

Calculate the Inception Score (IS) which is used to access how realistic generated images are.

$$IS = exp(\mathbb{E}_x KL(p(y | x ) || p(y)))
$$

where $KL(p(y | x) || p(y))$ is the KL divergence between the conditional distribution $p(y|x)$  
and the marginal distribution $p(y)$. Both the conditional and marginal distribution is calculated from features extracted from the images. The score is calculated on random splits of the images such that both a mean and standard deviation of the score are returned. The metric was originally proposed in [Imporved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
#### IS functional using case

```python
import mindspore as ms  
from mindone.metrics.functional import inception_score    
imgs = ms.ops.randint(0, 255, (100, 3, 299, 299), seed=123).to(ms.uint8)  
output = inception_score(imgs)  
print(f"IS: {output}")
# IS: (Tensor(shape=[], dtype=Float32, value= 1.0607), Tensor(shape=[], dtype=Float32, value= 0.0186892))
```

#### IS distributed using case

IS metric supports distributed environment for large dataset cases. Achieved with mindspore framework, you need to build an available mindspore distributed environment first. We currently only support rank table startup with ascend devices. Raise an issue if you prefer other ways or devices.

Firstly, generate a rank table json file to specify the device you prefer to use, guided by [mindspore official docs](https://www.mindspore.cn/tutorials/experts/en/r2.2/parallel/rank_table.html). You can generate rank table automatically with [hccl_tools](https://gitee.com/mindspore/models/blob/r1.7/utils/hccl_tools/hccl_tools.py).

Then write a python script for metric calculating:

```python
import mindspore as ms
from mindone.metrics.clip_directional_similarity import ClipDirectionalSimilarity
# set contexts, GRAPH_MODE is not supported
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
# init distributed environment for all gather operations
init()
# you could load in images and texts instead of generating random ones
imgs = ms.ops.randint(0, 255, (100, 3, 299, 299)).to(ms.uint8)
metric = InceptionScore()
output = metric(imgs)
print("IS: ", output)
```

At last, write a shell script to startup calculating (script below is a 4 device case), the python script above is named 'distributed_text.py'. Notice that you may prefer copying checkpoint_download folder to avoid downloading multi times.

```shell
#!/bin/bash

export DEVICE_NUM=4
export RANK_SIZE=4
export RANK_TABLE_FILE="/home/conf/hccl_4p_4567_127.0.0.1.json"

pids=()
for((i=4;i<8;i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./distributed_test.py ./device$i
    cp -R ./checkpoint_download ./device$i/checkpoint_download
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$(expr $i - 4)
    echo "start training for device $DEVICE_ID, rank id: $RANK_ID"
    env > env$i.log
    python ./distributed_test.py > test$i.log 2>&1 &
    cd ../
done

