# SDXL入门指南

本文提供了SDXL内置命令行工具的简要使用介绍。

## 依赖项

- MindSpore 2.2
- OpenMPI 4.0.3（用于分布式模式）

要安装依赖项，请运行

```shell
pip install -r requirements.txt

```

## 准备工作

### 转换预训练权重

我们提供了一个脚本，用于将预训练权重从`.safetensors`格式转换为`.ckpt`格式，位于`tools/model_conversion/convert_weight.py`。

步骤1. 从Hugging Face下载[官方](https://github.com/Stability-AI/generative-models)预训练权重 [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 和 [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)。

步骤2. 将权重转换为MindSpore的`.ckpt`格式并放入`./checkpoints/`。


```shell
cd tools/model_conversion

# convert sdxl-base-1.0 model
python convert_weight.py \
  --task pt_to_ms \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml

# convert sdxl-refiner-1.0 model
python convert_weight.py \
  --task pt_to_ms \
  --weight_safetensors /PATH TO/sd_xl_refiner_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_refiner_1.0_ms.ckpt \
  --key_torch torch_key_refiner.yaml \
  --key_ms mindspore_key_refiner.yaml
```

### 用于微调的数据集准备（可选）

进行微调的文本-图像数据集应该遵循以下文件结构


<details onclose>

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

img_txt.csv 是以下格式的注释文件
```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

为了方便起见，我们准备了两个公开的文本-图像数据集，符合上述格式。

- [Pokemon-Blip-Caption 数据集](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets)，包含 833 张 Pokemon 风格的图像，带有 BLIP 生成的标题。
- [Chinese-Art Blip Caption 数据集](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets)，包含 100 张中国艺术风格的图像，带有 BLIP 生成的标题。

要使用它们，请从 [openi 数据集网站](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets) 下载 `pokemon_blip.zip` 或 `chinese_art_blip.zip`，然后解压缩。

</details>


#### 使用 Webdataset 进行训练

图像文本配对数据被存档为 `tar` 文件在 Webdataset 中。一个训练数据集如下所示：

```text
data_dir
├── 00001.tar
│   ├── 00001.jpg
│   ├── 00001.json
│   ├── 00002.jpg
│   ├── 00002.json
│   └── ...
├── 00002.tar
├── 00003.tar
└── ...
```

我们提供了一个用于 Webdataset 的数据加载器 (`T2I_Webdataset_RndAcs`)，它与 MindData 的 GeneratorDataset 兼容。

1. 设置训练 YAML 配置如下以使用 T2I_Webdataset 加载器。
    ```yaml
        dataset_config:
            target: gm.data.dataset_wds.T2I_Webdataset_RndAcs
            params:
                caption_key: 'text_english'
    ```

2. 在训练脚本中设置 `--data_path`，指定整个训练数据集的根目录路径，例如上述示例中的 `data_dir`。

请注意，数据加载器是基于 [wids](https://github.com/webdataset/webdataset?tab=readme-ov-file#the-wids-library-for-indexed-webdatasets) 实现的。它需要一个 shardlist 信息文件，描述了每个 tar 文件的路径以及其中样本的数量。

shardlist 描述遵循以下格式。
```json
{
"__kind__": "wids-shard-index-v1",
"wids_version": 1,
"shardlist":
    [
        {"url": "data_dir/part01/00001.tar", "nsamples": 10000},
        {"url": "data_dir/part01/00002.tar", "nsamples": 10000},
    ]
}
```

首次运行时，数据加载器将扫描整个数据集以获取 shardlist 信息（对于大型数据集可能耗时较长），并将 shardlist 描述文件保存为 `{data_dir}/data_info.json`。后续运行时，数据加载器将重用现有的 `{data_dir}/data_info.json`，以节省扫描时间。

您可以通过配置 YAML 文件中的 `shardlist_desc` 参数手动指定新的 shardlist 描述文件，例如：

```yaml
    dataset_config:
        target: gm.data.dataset_wds.T2I_Webdataset_RndAcs
        params:
            caption_key: 'text_english'
            shardlist_desc: 'data_dir/data_info.json'
```

> 请注意，如果您已经更新了训练数据，则应该要么指定一个新的 shardlist 描述文件，要么**删除现有的 shardlist 文件** `{data_dir}/data_info.json` 以进行自动重新生成。

对于分布式训练，在使用 `T2I_Webdataset_RndAcs` 数据加载器时，无需额外的工作，因为它与 MindSpore 的 `GeneratorDataset` 兼容，并且数据分区将在 `GeneratorDataset` 中完成，就像使用原始数据格式进行训练一样。

## 推理

### 在线推理

我们提供了一个使用 [streamlit](https://streamlit.io/) 的文本到图像采样演示，可以在 `demo/sampling_without_streamlit.py` 和 `demo/sampling.py` 中找到。

在获得权重之后，请将它们放入 `checkpoints/` 目录中。然后，使用以下方法启动演示：


```shell
# (recommend) run with streamlit
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run demo/sampling.py --server.port <your_port>
```

> 如果在下载 clip tokenizer 时遇到网络问题，请手动从 huggingface 下载 `openai/clip-vit-large-patch14`，然后将 `configs/inference/sd_xl_base.yaml` 中的 `version: openai/clip-vit-large-patch14` 修改为 `version: your_path/to/clip-vit-large-patch14`。

2. 使用其他方法运行：


<details close>

```shell
# run sdxl-base txt2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend

# run sdxl-refiner img2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task img2img \
  --config configs/inference/sd_xl_refiner.yaml \
  --weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --img /PATH TO/img.jpg \
  --device_target Ascend

# run pipeline without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --add_pipeline True \
  --pipeline_config configs/inference/sd_xl_refiner.yaml \
  --pipeline_weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --sd_xl_base_ratios "1.0_768" \
  --device_target Ascend

# run lora(unmerge weight) without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt,SDXL-base-1.0_2000_lora.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend
```

</details>

<details>

  <summary>长文本支持</summary>

默认情况下，SD-XL 仅支持长度不超过 77 的令牌序列。对于长度超过 77 的序列，它们将被截断为 77，这可能导致信息丢失。

为了避免长文本提示的信息丢失，我们可以将一个长的令牌序列（N > 77）分成几个较短的子序列（N <= 77），以绕过文本编码器上下文长度的限制。`demo/sampling_without_streamlit.py` 中的 `args.support_long_prompts` 支持此功能。

在使用 `demo/sampling_without_streamlit.py` 进行推断时，您可以设置如下参数：


  ```bash
  python demo/sampling_without_streamlit.py \
  ...  \  # other arguments configurations
  --support_long_prompts True \  # allow long text prompts
  ```

在使用 `demo/sampling.py` 进行推断时，您只需输入您的长提示，并在提示下方点击“使用长文本提示支持（Token长度 > 77）”按钮，然后开始抽样。

</details>

### 离线推理

请参阅 [offline_inference](./offline_inference/README.md)。

### 使用 T2i-Adapter 进行推断

[T2I-Adapter](../t2i_adapter/README.md) 是一个简单轻量级的网络，为 Stable Diffusion 模型提供额外的视觉引导，无需重新训练。该适配器充当 SDXL 模型的插件，使其易于集成和使用。

有关使用 T2I-Adapter 进行推断和训练的更多信息，请参阅[T2I-Adapter](../t2i_adapter/README.md) 页面。

### 隐形水印检测

敬请期待

## 训练和微调

⚠️ 此功能属于实验性质。该脚本对整个模型进行微调，通常会导致模型过拟合并出现像灾难性遗忘等问题。建议尝试不同的超参数以获得数据集上的最佳结果。

我们在 `configs/training` 中提供了示例训练配置。要启动训练，请执行以下操作：

1. 普通微调，示例如下：

```shell
# sdxl-base fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \

# sdxl-base fine-tune with 8p on Ascend
mpirun --allow-run-as-root -n 8 python train.py \
  --config configs/training/sd_xl_base_finetune_multi_graph_910b.yaml \
  --weight "" \
  --data_path /PATH TO/YOUR DATASET/ \
  --max_device_memory "59GB" \
  --param_fp16 True \
  --is_parallel True

# sdxl-base fine-tune with 16p on Ascend
bash scripts/run_vanilla_ft_910b_16p /path_to/hccl_16p.json 0 8 16 /path_to/dataset/  # run on server 1
bash scripts/run_vanilla_ft_910b_16p /path_to/hccl_16p.json 8 16 16 /path_to/dataset/ # run on server 2
```

2. LoRA微调, 例如:

```shell
# sdxl-base lora fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --gradient_accumulation_steps 4 \
```

3. DreamBooth微调:

详情请参阅 [dreambooth_finetune.md](./dreambooth_finetune.md).

4. 文本反演微调

详情请参阅 [textual_inversion_finetune.md](./textual_inversion_finetune.md)。

5. 使用多个NPU运行，示例如下：


```shell
# run with multiple NPU/GPUs
mpirun --allow-run-as-root -n 8 python train.py \
  --config /PATH TO/config.yaml \
  --weight /PATH TO/weight.ckpt \
  --data_path /PATH TO/YOUR DATASET/ \
  --is_parallel True \
  --device_target <YOUR DEVCIE>
```
