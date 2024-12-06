# LLaVA-NeXT: Open Large Multimodal Models (MindSpore)

This repo contains Mindspore model definitions, pre-trained weights and inference/sampling code for the [model](https://llava-vl.github.io/blog/2024-01-30-llava-next/). Referring to the [official project page](https://github.com/LLaVA-VL/LLaVA-NeXT).

## Dependencies and Installation

- CANN: 8.0.RC2 or later
- Python: 3.9 or later
- Mindspore: 2.3.1

## Getting Start

### Downloading Pretrained Checkpoints

Please download the model [llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) to the `./models` directory. And run

```bash
python tools/convert_llava.py models/llava-v1.6-mistral-7b-hf -o models/llava-v1.6-mistral-7b-hf/model.ckpt
```

to convert the model weight in Mindspore `ckpt` format.

### Inference

To run the inference, you may use `predict.py` with the following command

```bash
python predict.py --input_image path_to_your_input_image --prompt input_prompt
```

For example, running `python predict.py` with the default image [llava_v1_5_radar.jpg](https://github.com/user-attachments/assets/8e016871-82fd-488a-8629-5ca71222e0e3) and default prompt `What is shown in this image?` will give the following result:

```text
[INST]
What is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multivariate chart that displays values for multiple variables represented on axes
starting from the same point. This particular radar chart is showing the performance of different models or systems across various metrics.

The axes represent different metrics or benchmarks, such as MM-Vet, MM-Vet, MM-Vet, MM-Vet, MM-Vet, MM-V
```

## Benchmark

### Inference

To perform the benchmark, you may first download the image [llava_v1_5_radar.jpg](https://github.com/user-attachments/assets/8e016871-82fd-488a-8629-5ca71222e0e3) and save it in `./assets`, and then run `python predict --benchmark` to get the throughput.

|         Model         | Context       | Batch Size | Throughput (tokens/second)|
|-----------------------|---------------|------------|---------------------------|
| llava-v1.6-mistral-7b | D910*x1-MS2.3 |    1       | 21.2                      |

> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.\
> Throughput (tokens/second): number of generated tokens per second.\
> We use the second round of inference as the benchmark result.
