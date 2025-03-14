# JanusPro Training

## Requirements

Please install the dependencies and download the pre-trained models according to the instructions in [README.md](../README.md).

## Dataset Preparation

| Task      |  Huggingface Dataset Link |
| :----------     |    :------        |
| pure text     |     [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA/tree/main)     |
| text-to-image           |     [jasonhuang23/artwork](https://huggingface.co/datasets/jasonhuang23/artwork)      |
| VQA           |     [rbojja/medical-vqa](https://huggingface.co/datasets/rbojja/medical-vqa/tree/main)      |


We will use the above datasets for JanusPro fine-tuning. Please download them by:

```shell
huggingface-cli download  qiaojin/PubMedQA --repo-type dataset --local-dir datasets/PubMedQA
huggingface-cli download  jasonhuang23/artwork --repo-type dataset --local-dir datasets/artwork
huggingface-cli download  rbojja/medical-vqa --repo-type dataset --local-dir datasets/medical-vqa
```

## Run Training

- Text Generation Task

```shell
bash scripts/run_sft_text.sh
```

- Text-to-Image Generation Task (T2I)

```shell
bash scripts/run_sft_t2i.sh
```

- Multimodal Understanding Task (VQA)

```shell
bash scripts/run_sft_vqa.sh
```

The default training stage is stage 3, that is, all modules are trainable except for VQ16 for image token decoding. To switch to other stage, you can modify the `--stage` argument in the training script.

For more detailed arguments, please run `python train.py -h`.


- Multi-task Fune-tuning

Comming soon



## Performance

Experiments are tested on ascend 910* with mindspore 2.5.0 pynative mode:

| model | task | # card(s) | image size | max_length | batch size | step time (s/step)|
|:-:|:--:| :-:|:-:|:-:|:-:|:-:|
| Janus-Pro-1B | T2I | 1 | 384x384 | 1024   | 8 | 0.66 |
| Janus-Pro-1B | VQA | 1 | 384x384 | 1024   | 4 | 0.59 |
| Janus-Pro-1B | Text | 1 | n.a. | 512   | 8 | 0.50 |
| Janus-Pro-7B | T2I | 1 | 384x384 | 1024   | 1 | 0.49 |
| Janus-Pro-7B | VQA | 1 | 384x384 | 1024   | 1 |  0.66 |
| Janus-Pro-7B | Text | 1 | n.a. | 512   | 1 | 0.53 |
