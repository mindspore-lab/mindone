# FAQ - Frequently Asked Questions

### 1、如何使用相同的权重，达成在 MindSpore 910* 上推理结果与 PyTorch [Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README_sdxl.md) A100 一致

<details onclose>

#### 回答：

#### (1) 权重转换

- diffuser中openclip pool层 text_projection 操作实现与官方stability-ai不一致，本处follow官方实现； text_projection在两边实现中相差了一个转置操作，自行转换的时候需要检查注意下，本处提供的模型转换[小工具](../tools/model_conversion/README.md)中已带有[转置操作](https://github.com/mindspore-lab/mindone/blob/468be76a35e9308c7b59bf9eaf3791c146539ee4/examples/stable_diffusion_xl/tools/model_conversion/convert_diffusers_to_mindone_sdxl.py#L306C9-L310C48)

#### (2) 超参配置

- 默认超参保持一致 (sampler、sample step、noise input、cfg 等)

- ms中配置以下参数 (以 EulerAncestralSampler 为例):
    - `--discretization DiffusersDDPMDiscretization`
    - `--precision_keep_origin_dtype True`

    ```shell
    python demo/sampling_without_streamlit.py \
      --config configs/inference/sd_xl_base.yaml \
      --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
      --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
      --sampler EulerAncestralSampler \
      --sample_step 40 \
      --guidance_scale 5.0 \
      --discretization DiffusersDDPMDiscretization \
      --precision_keep_origin_dtype True
    ```

#### (3) 固定输入noise

- 在diffusers运行过程中保存输入latent为`.npy文件`, 通过 `--init_latent_path` 接口控制输入latent noise 固定输入噪声一致；

- (可选, 仅带随机采样的sampler需要配置) 在diffusers运行过程中保存每个sample step添加的随机噪声为`.npy文件`, 通过 `--init_noise_scheduler_path` 配置固定每个sample step添加的随机噪声一致，当前仅`EulerA采样器`生效；

</details>


### 2、如何使用相同的权重，达成在 MindSpore 910* 上推理结果与 PyTorch [Stability-AI](https://github.com/Stability-AI/generative-models) A100 一致

<details onclose>

#### 回答：

#### (1) 超参配置

- 默认超参保持一致 (sampler、sample step、noise input、cfg 等)

    ```shell
    python demo/sampling_without_streamlit.py \
      --config configs/inference/sd_xl_base.yaml \
      --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
      --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
    ```

#### (2) 固定输入noise

- 在stability-ai代码运行过程中保存输入latent为`.npy文件`, 通过 `--init_latent_path` 接口控制输入latent noise 固定输入噪声一致；

</details>


### 3、如何达成在 MindSpore 910* 上 vanilla finetune 训练结果与 PyTorch [Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README_sdxl.md) A100 相近

<details onclose>

#### 回答：

#### (1) 预训练权重、训练数据集 保持一致

#### (2) 代码实现保持一致，均使用公版DDPM流程进行训练

#### (3) 超参数保持一致，超参配置及介绍参考[hyper_parameters.md](./hyper_parameters.md)

</details>


### 4、如何使用 fp16 权重进行训练 (不推荐)

<details onclose>

#### 回答：

#### (1) 通过 `--param_fp16 True` 进行配置，在cache data的场景下，PerBatchSize 达到 8;

#### ⚠️注意：`--param_fp16` 是实验性参数，打开后可能会导致训练不稳定、崩溃，请谨慎使用；

#### ⚠️注意：在了解了风险之后如果仍然希望使用`fp16`权重进行训练，那么可以配合 fp32 vae [cache](./vanilla_finetune.md) 使用 或 配合[vae-fp16-fix](./weight_convertion.md)权重使用

</details>


### 5. 连接不上huggingface, 报错 `Can't load tokenizer for 'openai/clip-vit-large-patch14'`

<details onclose>

#### 回答：

#### 该问题是因为网络原因无法连接到 huggingface，可以尝试手动下载后指定本地路径解决, [issue 134](https://github.com/mindspore-lab/mindone/issues/134)；

#### (1) 在huggingface上下载[clip patch](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)

#### (2) 在config(.yaml) 文件中 `version:` 参数指定本地路径:

  ```shell
  model:
    target: gm.models.diffusion.DiffusionEngine
    params:
      ...
      conditioner_config:
        target: gm.modules.GeneralConditioner
        params:
          emb_models:
            - is_trainable: False
              input_key: txt
              target: gm.modules.embedders.modules.FrozenCLIPEmbedder
              params:
                layer: hidden
                layer_idx: 11
                version: /path/to/clip-vit-large-patch14
            - ...
  ```

</details>

### 6. 在Ascend 910* 可以正常使用flash attention，但在Ascend 910使用时报错。
<details onclose>

#### 回答：

#### Mindspore 2.2 之后的版本，框架内的 `FlashAttention` 只支持Ascend 910*， 不支持Ascend 910。如果需要在 Ascend 910 上使用 flash attention，可以借助基于 Mindspore 框架和昇腾 CANN 软件栈开发的 transformer 加速库 [acctransformer](https://github.com/mindspore-ai/acctransformer/tree/master/train/acctransformer/flash_attention)。具体使用方法如下：

#### (1) 按照 acctransformer 的 [FlashAttention2安装指南](https://github.com/mindspore-ai/acctransformer/tree/master/train/acctransformer/flash_attention)安装whl包。

  ```shell
  git clone https://github.com/mindspore-ai/acctransformer.git
  cd train
  bash build.sh
  pip install dist/acctransformer-1.0.0-py3-none-any.whl
  ```

#### (2) 把examples/stable_diffusion_xl/gm/modules/attention.py文件 Mindspore 框架内的 `FlashAttention` 改成 acctransformer 的，具体修改如下:

https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl/gm/modules/attention.py#L18
https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl/gm/modules/attention.py#L117
https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl/gm/modules/attention.py#L147


  ```diff
  # L18
  - from mindspore.nn.layer.flash_attention import FlashAttention
  + from acctransformer.flash_attention.nn.layer.flash_attention import FlashAttention
  ...

  # L117
  - self.flash_attention = FlashAttention(head_dim=dim_head, head_num=heads, high_precision=True)
  + self.flash_attention = FlashAttention(head_dim=dim_head)

  # L147
  - out = self.flash_attention(q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(ms.uint8))
  + out = self.flash_attention(q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(ms.float16))

  ```

</details>
