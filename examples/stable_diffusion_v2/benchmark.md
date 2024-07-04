# Stable Diffusion Benchmark

## Text-to-Image

### Training

| SD Model      |   Context      |  Method      | Global Batch Size x Grad. Accu. |   Resolution       |   Acceleration   | jit_level  |FPS (img/s)  |
|---------------|---------------|--------------|:-------------------:|:------------------:|:----------------:|:----------------:|----------:|
| 1.5           |    D910x1-MS2.1      |    Vanilla   |      3x1             |     512x512         | Graph, DS, FP16,  |     N/A |       5.98          |
| 1.5           |    D910x8-MS2.1      |    Vanilla   |      24x1             |     512x512         | Graph, DS, FP16,  |    N/A |     31.18          |
| 1.5           |    D910x1-MS2.1      |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |      N/A |   8.25          |
| 1.5           |    D910x8-MS2.1      |    LoRA      |      32x1             |     512x512         | Graph, DS, FP16,  |      N/A |  63.85          |
| 1.5           |    D910x1-MS2.1      |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |   N/A |   2.09            |
| 2.0           |    D910x1-MS2.1       |    Vanilla      |      3x1             |     512x512         | Graph, DS, FP16,  |     N/A |   6.19          |
| 2.0           |    D910x8-MS2.1       |    Vanilla      |      24x1             |     512x512         | Graph, DS, FP16,  |     N/A |   33.50          |
| 2.0           |    D910x1-MS2.1       |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |       N/A |   9.46       |
| 2.0           |    D910x8-MS2.1       |    LoRA      |      32x1             |     512x512         | Graph, DS, FP16,  |      N/A |    73.51       |
| 2.0           |    D910x1-MS2.1       |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |     N/A |  2.18            |
| 2.1-v           |    D910x1-MS2.1       |    Vanilla      |      3x1             |     768x768         | Graph, DS, FP16, FA  |      N/A |   3.16          |
| 2.1-v           |    D910x8-MS2.1       |    Vanilla      |      24x1             |     768x768         | Graph, DS, FP16, FA  |     N/A |    18.98          |
| 2.1-v           |    D910x1-MS2.1       |    LoRA      |      4x1                 |     768x768         | Graph, DS, FP16, FA  |      N/A |   3.39          |
| 2.1-v           |    D910x8-MS2.1       |    LoRA      |      32x1                 |     768x768         | Graph, DS, FP16, FA  |     N/A |    23.45          |
| 1.5           |    D910*x1-MS2.3      |    Vanilla   |      3x1             |     512x512         | Graph, DS, FP16,  |     O2 |   11.86          |
| 1.5           |    D910*x8-MS2.3      |    Vanilla   |      24x1             |     512x512         | Graph, DS, FP16,  |  O2 |  75.53          |
| 1.5           |    D910*x1-MS2.3      |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |   O2 |  15.27          |
| 1.5           |    D910*x8-MS2.3      |    LoRA      |      32x1             |     512x512         | Graph, DS, FP16,  |   O2 |  119.94          |
| 1.5           |    D910*x1-MS2.3      |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |  O2 |    3.86          |
| 2.0           |    D910*x1-MS2.3       |    Vanilla      |      3x1             |     512x512         | Graph, DS, FP16,  |     O2 |   12.75        |
| 2.0           |    D910*x8-MS2.3       |    Vanilla      |      24x1             |     512x512         | Graph, DS, FP16,  |   O2 |    79.67        |
| 2.0           |    D910*x1-MS2.3       |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |      O2 |     16.53     |
| 2.0           |    D910*x8-MS2.3       |    LoRA      |      32x1             |     512x512         | Graph, DS, FP16,  |     O2 |      129.70     |
| 2.0           |    D910*x1-MS2.3       |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |  O2 |   3.76         |
| 2.1-v           |    D910*x1-MS2.2.10       |    Vanilla      |      3x1             |     768x768         | Graph, DS, FP16,  |   N/A |   5.80        |
| 2.1-v           |    D910*x1-MS2.2.10       |    Vanilla      |      24x1             |     768x768         | Graph, DS, FP16,  |   N/A |    46.02        |
| 2.1-v           |    D910*x1-MS2.2.10       |    LoRA      |      4x1                 |     768x768         | Graph, DS, FP16,  |   N/A |   6.65        |
| 2.1-v           |    D910*x8-MS2.2.10       |    LoRA      |      32x1                 |     768x768         | Graph, DS, FP16,  |   N/A |    52.57        |
> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
>
> Acceleration: DS: data sink mode, FP16: float16 computation. FA: flash attention.
>
>FPS: images per second during training. average training time (s/step) = batch_size / FPS
>
>jie_level: Used to control the compilation optimization level. N/A means that the current MindSpore version does not support setting jit_level.

Note that the jit_level only can be used for MindSpore 2.3.

Note that the performance of SD2.1 should be similar to SD2.0 since they have the same network architecture.

Note that SD1.x and SD2.x share the same UNet architecture, thus their performance on vanilla training are similar.

<!--
TB tested:
| 1.5           |    D910x1-MS2.1      |    ControlNet      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |
| 2.1-v           |    D910x1-MS2.1       |    Dreambooth      |      1x1             |     768x768         | Graph, DS, FP16,  |                 |
| 1.5           |    D910*x1-MS2.2      |    ControlNet      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |

Other Acceleration techniques:
Flash Attention,
-->


### Inference

| SD Model      |     Context |  Scheduler   | Steps              |  Resolution   |      Batch Size     |  Speed (step/s)     | FPS (img/s)     |
|---------------|:-----------|:------------:|:------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| 1.5           |     D910x1-MS2.2.10    |  DDIM       |   30       |    512x512         |       4          |    3.58        |       0.44       |
| 2.0           |     D910x1-MS2.2.10    |  DDIM       |   30       |    512x512         |       4          |    4.12       |        0.49       |
| 2.1-v         |     D910x1-MS2.2.10    |  DDIM       |   30       |    768x768         |       4          |     1.14     |         0.14      |
| 1.5           |     D910*x1-MS2.2.10   |  DDIM       |   30       |    512x512         |       4          |       6.19     |         0.71      |
| 2.0           |     D910*x1-MS2.2.10   |  DDIM       |   30       |    512x512         |       4          |      7.65     |         0.83      |
| 2.1-v         |     D910*x1-MS2.2.10   |  DDIM       |   30       |    768x768         |       4          |      2.79     |         0.32      |
> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
>
> Speed (step/s): sampling speed measured in the number of sampling steps per second.
>
> FPS (img/s): image generation throughput measured in the number of image generated per second.

Note that the performance of SD2.1 should be similar to SD2.0 since they have the same network architecture. Performance per NPU in multi-NPU parallel mode is the same as performance of single NPU mode.


<!--
Add a column for model/pipeline yaml config?
Mixed precision belongs to configuration
-->

## Image-to-Image

Coming soon
