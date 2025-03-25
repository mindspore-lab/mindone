# Download Pretrained Models

All models are stored in `hunyuanvideo-i2v/ckpts` by default, and the file structure is as follows
```shell
hunyuanvideo-i2v
  ├──ckpts
  │  ├──README.md
  │  ├──hunyuan-video-i2v-720p
  │  │  ├──transformers
  │  │  │  ├──mp_rank_00_model_states.pt
  ├  │  ├──vae
  ├  │  ├──lora
  │  │  │  ├──embrace_kohaya_weights.safetensors
  │  │  │  ├──hair_growth_kohaya_weights.safetensors
  │  ├──text_encoder_i2v
  │  ├──text_encoder_2
  ├──...
```

## Download HunyuanVideo-I2V model
To download the HunyuanVideo-I2V model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'HunyuanVideo-I2V'
cd HunyuanVideo-I2V
# Use the huggingface-cli tool to download HunyuanVideo-I2V model in HunyuanVideo-I2V/ckpts dir.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts
```

<details>
<summary>💡Tips for using huggingface-cli (network problem)</summary>

##### 1. Using HF-Mirror

If you encounter slow download speeds in China, you can try a mirror to speed up the download process. For example,

```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts
```

##### 2. Resume Download

`huggingface-cli` supports resuming downloads. If the download is interrupted, you can just rerun the download
command to resume the download process.

Note: If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download
process, you can ignore the error and rerun the download command.

</details>

---

## Download Text Encoder

HunyuanVideo-I2V uses an MLLM model and a CLIP model as text encoder.

1. MLLM model (text_encoder_i2v folder)

HunyuanVideo-I2V supports different MLLMs (including HunyuanMLLM and open-source MLLM models). At this stage, we have not yet released HunyuanMLLM. We recommend the user in community to use [llava-llama-3-8b](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers) provided by [Xtuer](https://huggingface.co/xtuner), which can be downloaded by the following command.

Note that unlike [HunyuanVideo](https://github.com/Tencent/HunyuanVideo/tree/main), which only uses the language model parts of `llava-llama-3-8b-v1_1-transformers`, HunyuanVideo-I2V needs its full model to encode both prompts and images. Therefore, you only need to download the model without preprocessing.

```shell
cd HunyuanVideo-I2V/ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./text_encoder_i2v
```

2. CLIP model (text_encoder_2 folder)

We use [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) provided by [OpenAI](https://openai.com) as another text encoder, users in the community can download this model by the following command

```
cd HunyuanVideo-I2V/ckpts
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
```

## Model Conversion

To convert the vae checkpoint, please run the following command:
```bash
python tools/convert_pytorch_ckpt_to_safetensors.py --src ckpts/hunyuan-video-i2v-720p/vae/pytorch_model.pt --target ckpts/hunyuan-video-i2v-720p/vae/model.safetensors --config  ckpts/hunyuan-video-i2v-720p/vae/config.json
```
