# TODOLIST

- [ ] 模型代码迁移
- [ ] 推理代码迁移
- [ ] 性能对比
- [ ] 精度对比
- [ ] 预处理数据代码迁移
- [ ] 训练流程

# 模型结构

![framework](./figures/framework.png)

1、多模态编码层：

​	imageBind进行编码，

​	映射层映射到大模型可理解的形式

2、大模型推理

​	Vicuna 

3、多模态生成阶段

​	接收来自LLM的指令，输出投影层将信号令牌表示映射成多模态解码器可以理解的表示

​	使用现成的潜码条件扩散模型生成图像、视频和音频（SD）

​	image : stable diffuse

​	Audio: AudioLDM

​	Video Diffusion: ZeroScope



# 模型依赖

- LLama

- ImageBind

- SD1.5

  ```
  from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
  from diffusers.configuration_utils import FrozenDict
  from diffusers.models import AutoencoderKL, UNet2DConditionModel
  from diffusers.schedulers import KarrasDiffusionSchedulers
  from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
  from diffusers.pipeline_utils import DiffusionPipeline
  from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
  from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
  ```

- AudioLDM

  ```
  from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan
  
  from diffusers.models import AutoencoderKL, UNet2DConditionModel
  from diffusers.schedulers import KarrasDiffusionSchedulers
  from diffusers.utils import is_accelerate_available, logging, randn_tensor, replace_example_docstring
  from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
  ```

- ZeroScope

  ```
  from transformers import CLIPTextModel, CLIPTokenizer
  
  from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
  from diffusers.models import AutoencoderKL, UNet3DConditionModel
  from diffusers.schedulers import KarrasDiffusionSchedulers
  from diffusers.utils import (
      is_accelerate_available,
      is_accelerate_version,
      logging,
      randn_tensor,
      replace_example_docstring,
  )
  from diffusers.pipeline_utils import DiffusionPipeline
  from diffusers.utils import BaseOutput
  ```

- qFormer

  ```
  from transformers.activations import ACT2FN
  from transformers.file_utils import (
      ModelOutput,
  )
  from transformers.modeling_outputs import (
      BaseModelOutputWithPastAndCrossAttentions,
      BaseModelOutputWithPoolingAndCrossAttentions,
      CausalLMOutputWithCrossAttentions,
      MaskedLMOutput,
      MultipleChoiceModelOutput,
      NextSentencePredictorOutput,
      QuestionAnsweringModelOutput,
      SequenceClassifierOutput,
      TokenClassifierOutput,
  )
  from transformers.modeling_utils import (
      PreTrainedModel,
      apply_chunking_to_forward,
      find_pruneable_heads_and_indices,
      prune_linear_layer,
  )
  from transformers.utils import logging
  from transformers.models.bert.configuration_bert import BertConfig
  ```

- NEXTGPT

  ```
  from transformers import StoppingCriteria, StoppingCriteriaList
  ```

​		

# 模型文件：

- `ImageBind`
is the unified image/video/audio encoder. The pre-trained checkpoint can be downloaded from [here](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) with version `huge`. Afterward, put the `imagebind_huge.pth` file at [[./ckpt/pretrained_ckpt/imagebind_ckpt/huge]](ckpt/pretrained_ckpt/imagebind_ckpt/). 
- `Vicuna`:
first prepare the LLaMA by following the instructions [[here]](ckpt/pretrained_ckpt/prepare_vicuna.md). Then put the pre-trained model at [[./ckpt/pretrained_ckpt/vicuna_ckpt/]](ckpt/pretrained_ckpt/vicuna_ckpt/). 
- `Image Diffusion`
is used to generate images. NExT-GPT uses [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) with version `
v1-5`. (_will be automatically downloaded_)
- `Audio Diffusion`
for producing audio content. NExT-GPT employs [AudioLDM](https://github.com/haoheliu/AudioLDM) with version `l-full`. (_will be automatically downloaded_)
- `Video Diffusion`
for the video generation. We employ [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) with version `v2_576w`. (_will be automatically downloaded_)

