# Tango
A text-to-audio pipeline, based on stable-diffusion. Paper: [Text-to-Audio Generation using Instruction Tuned LLM and Latent Diffusion Model](https://arxiv.org/abs/2304.13731).

Some domain knowledge on audio processing:

- raw audio data is of shape `[batch, time, channels]`, e.g., `[1, 16000, 1]` for 1 second of 16kHz mono audio.

- a common preprocessing technique on raw audio is to compute mel-spectrograms to get features of shape: `[batch, 1, time // hop_length, num_mels]`, e.g., `hop_length = 320, num_mels=80`.

- now everything basically follows stable-diffusion for text-to-image!

- the text encoder used for conditioning is flan-t5-large, which gives `c` of shape `[batch, max_length=512, dim=1024]`.

## Demo

"crowd applauding clapping hands" [audio](https://github.com/genshimamber/mindone/assets/145047261/b8537df0-4ba1-49a4-b961-0a15392b6ca7)

"a dog is barking" [audio](https://github.com/genshimamber/mindone/assets/145047261/f67cf926-d0e7-43b4-a690-2f05eb7a96db)

"a cat is meowing" [audio](https://github.com/genshimamber/mindone/assets/145047261/647d7834-5ff5-4e70-96a7-d9ae718a0285)

Note: we manually converted the generated recordings from `.wav` to `.webm` so we could put them in readme.

## Getting Started

1. download weights from [tango_full_ft_audiocaps-fa8f707f](https://download.mindspore.cn/toolkits/mindone/tango/tango_full_ft_audiocaps-fa8f707f.ckpt). Ref: [tango-full-ft-audiocaps](https://huggingface.co/declare-lab/tango-full-ft-audiocaps).

2. run:

```shell
python text_to_audio.py \
  --prompts "A dog is barking" \
  --config_path "configs" \
  --ckpt tango_full_ft_audiocaps-fa8f707f.ckpt \
  --num_steps 200 \
  --batch_size 1 \
  --guidance 3 \
  --num_samples 1

```

## Training

### Data

### Full-Train

### LoRA

## Evaluation

## Acknowledgements

## License
