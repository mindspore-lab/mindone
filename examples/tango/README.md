# Tango
A text-to-audio pipeline, based on stable-diffusion. Some domain knowledge on audio:

- raw audio data is of shape [batch, time, channels], where time is Hz * seconds, channels is 1 for mono or 2 for stereo.

- a common preprocessing technique on raw audio is to compute mel-spectrograms, giving features of shape: [batch, 1, time // hop_length, num_mels], where hop_length is usually 320, and num_mels is usually 80.

- then everything basically follows stable-diffusion for text-to-image!

- text encoder: flan-t5-large that gives c of shape [batch, length=512, dim=1024]

## Demo

"A dog is barking" [audio]()

"A cat is meowing" [audio]()

"Playing Chopin Piano Concerto 1" [audio]()

## Getting Started

1. download [ckpt]()

2. run:

```shell
python text_to_audio.py \
--prompts "A dog is barking" \
--config_path "configs" \
--ckpt "PATH_TO_CKPT" \
--num_steps 200 \
--batch_size 1 \
--guidance 3 \
--num_samples 1 \
```

## Training

### Data

### Train

## Acknowledgements

## License
