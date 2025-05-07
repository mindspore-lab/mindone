# PLLaVA based on MindSpore

MindSpore implementation of
[PLLaVA : Parameter-free LLaVA Extension from Images to Videos for Video Dense Captioning
](https://arxiv.org/abs/2404.16994).

## Requirements

| mindspore | ascend driver |    firmware    | cann toolkit/kernel |
|:---------:|:-------------:|:--------------:|:-------------------:|
|  2.4.1    |   24.1.RC3    | 7.5.T11.0.B088 |    8.0.RC3.beta1    |

## Getting Started
### Downloading Pretrained Checkpoints

Please download the model [here](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf).
By default, you can put the model under `./models` or your designated directory.

```bash
python tools/convert_pllava.py models/pllava7b/ -o models/pllava7b/model.ckpt
```

to convert the model weight in Mindspore `ckpt` format.

### Requirements

Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Inference

To run the inference, you may use `pllavarun.py` with the following command:

```bash
python pllavarun.py --video path_to_your_video
```

The inference examples are shown below:

| Video ID | Caption                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <video src="https://github.com/user-attachments/assets/e79c8b19-b5f6-4391-8bf4-4921e2fede15" /> | The image shows a collection of cake pans inside an oven. The pans are arranged on a rack, and each pan contains a different color of cake batter. The colors range from purple to yellow, indicating that the cakes are being baked in a single oven, likely in a batch process. The oven appears to be preheating, as suggested by the lighting and the presence of the cake batter. This scene is typical of a bakery or home kitchen where cakes are being prepared for baking. |


## Benchmark

### Inference

Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.4.1 PyNative mode.

To test the benchmark, you may use the video `-0og5HrzhpY_0.mp4` and place it under `./examples`
and run the following command:

```bash
python pllavarun.py --video ./example/-0og5HrzhpY_0.mp4 --benchmark
```

| model name | cards | jit level | batch size | throughput (tokens/second) |
|------------|-------|---------|------------|----------------------------|
| pllava-7b  | 1     | O1      | 1          | 16.2                       |

> We use the second round of inference as the benchmark result.
