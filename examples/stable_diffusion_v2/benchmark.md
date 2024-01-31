# Stable Diffusion Benchmark

## Text-to-Image

### Training

| SD Model      |   Context      |  Method      | Global Batch Size x Grad. Accu. |   Resolution       |   Acceleration   |   FPS (img/s)  |
|---------------|---------------|--------------|:-------------------:|:------------------:|:----------------:|:----------------:|
| 1.5           |    D910x1-MS2.1      |    Vanilla   |      3x1             |     512x512         | Graph, DS, FP16,  |       5.98          |
| 1.5           |    D910x8-MS2.1      |    Vanilla   |      24x1             |     512x512         | Graph, DS, FP16,  |       31.18          |
| 1.5           |    D910x1-MS2.1      |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |       8.25          |
| 1.5           |    D910x8-MS2.1      |    LoRA      |      32x1             |     512x512         | Graph, DS, FP16,  |       63.85          |
| 1.5           |    D910x1-MS2.1      |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |     2.09            |
| 2.0           |    D910x1-MS2.1       |    Vanilla      |      3x1             |     512x512         | Graph, DS, FP16,  |       6.19          |
| 2.0           |    D910x8-MS2.1       |    Vanilla      |      24x1             |     512x512         | Graph, DS, FP16,  |       33.50          |
| 2.0           |    D910x1-MS2.1       |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |          9.46       |
| 2.0           |    D910x8-MS2.1       |    LoRA      |      32x1             |     512x512         | Graph, DS, FP16,  |          73.51       |
| 2.0           |    D910x1-MS2.1       |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |     2.18            |
| 2.1-v           |    D910x1-MS2.1       |    Vanilla      |      3x1             |     768x768         | Graph, DS, FP16, FA  |       3.16          |
| 2.1-v           |    D910x8-MS2.1       |    Vanilla      |      24x1             |     768x768         | Graph, DS, FP16, FA  |       18.98          |
| 2.1-v           |    D910x1-MS2.1       |    LoRA      |      4x1                 |     768x768         | Graph, DS, FP16, FA  |       3.39          |
| 2.1-v           |    D910x8-MS2.1       |    LoRA      |      32x1                 |     768x768         | Graph, DS, FP16, FA  |       23.45          |
| 1.5           |    D910*x1-MS2.2.10      |    Vanilla   |      3x1             |     512x512         | Graph, DS, FP16,  |       9.22          |
| 1.5           |    D910*x8-MS2.2.10      |    Vanilla   |      24x1             |     512x512         | Graph, DS, FP16,  |      52.30          |
| 1.5           |    D910*x1-MS2.2.10      |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |       13.58          |
| 1.5           |    D910*x8-MS2.2.10      |    LoRA      |      32x1             |     512x512         | Graph, DS, FP16,  |       105.08          |
| 1.5           |    D910*x1-MS2.2.10      |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |       2.92          |
| 2.0           |    D910*x1-MS2.2.10       |    Vanilla      |      3x1             |     512x512         | Graph, DS, FP16,  |         10.03        |
| 2.0           |    D910*x8-MS2.2.10       |    Vanilla      |      24x1             |     512x512         | Graph, DS, FP16,  |         55.69        |
| 2.0           |    D910*x1-MS2.2.10       |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |            15.88     |
| 2.0           |    D910*x8-MS2.2.10       |    LoRA      |      32x1             |     512x512         | Graph, DS, FP16,  |            119.74     |
| 2.0           |    D910*x1-MS2.2.10       |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |        2.93         |
| 2.1-v           |    D910*x1-MS2.2.10       |    Vanilla      |      3x1             |     768x768         | Graph, DS, FP16,  |         5.80        |
| 2.1-v           |    D910*x1-MS2.2.10       |    Vanilla      |      24x1             |     768x768         | Graph, DS, FP16,  |         46.02        |
| 2.1-v           |    D910*x1-MS2.2.10       |    LoRA      |      4x1                 |     768x768         | Graph, DS, FP16,  |         6.65        |
| 2.1-v           |    D910*x8-MS2.2.10       |    LoRA      |      32x1                 |     768x768         | Graph, DS, FP16,  |         52.57        |
> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
>
> Acceleration: DS: data sink mode, FP16: float16 computation. FA: flash attention.
>
>FPS: images per second during training. average training time (s/step) = batch_size / FPS

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
| 1.5           |     D910x1-MS2.0    |  DPM++       |   20       |    512x512         |       4          |    2.50        |           0.40   |
| 2.0           |     D910x1-MS2.0    |  DPM++       |   20       |    512x512         |       4          |    2.86       |        0.44       |
| 2.1-v         |     D910x1-MS2.0    |  DPM++       |   20       |    768x768         |       4          |     1.18      |         0.19      |
| 1.5           |     D910*x1-MS2.2   |  DPM++       |   20       |    512x512         |       4          |       2.50     |         0.39      |
| 2.0           |     D910*x1-MS2.2   |  DPM++       |   20       |    512x512         |       4          |      2.86     |         0.42      |
| 2.1-v         |     D910*x1-MS2.2   |  DPM++       |   20       |    768x768         |       4          |      1.67     |         0.25      |
> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
> Speed (step/s): sampling speed measured in the number of sampling steps per second.
> FPS (img/s): image generation throughput measured in the number of image generated per second.

Note that the performance of SD2.1 should be similar to SD2.0 since they have the same network architecture.


<!--
Add a column for model/pipeline yaml config?
Mixed precision belongs to configuration
-->

## Image-to-Image

Coming soon
