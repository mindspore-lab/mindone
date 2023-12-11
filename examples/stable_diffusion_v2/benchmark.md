# Stable Diffusion Benchmark

## Text-to-Image

### Training

| SD Model      |   Context      |  Method      | Batch Size x Grad. Accu. |   Resolution       |   Acceleration   |   FPS (img/s)  |
|---------------|---------------|--------------|:-------------------:|:------------------:|:----------------:|:----------------:|
| 1.5           |    D910x1-MS2.1      |    Vanilla   |      3x1             |     512x512         | Graph, DS, FP16,  |       5.98          |
| 1.5           |    D910x1-MS2.1      |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |       8.25          |
| 1.5           |    D910x1-MS2.1      |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |     2.09            |
| 2.0           |    D910x1-MS2.1       |    Vanilla      |      3x1             |     512x512         | Graph, DS, FP16,  |       7.21          |
| 2.0           |    D910x1-MS2.1       |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |          8.87       |
| 2.0           |    D910x1-MS2.1       |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |     2.18            |
| 2.1-v           |    D910x1-MS2.1       |    Vanilla      |      3x1             |     768x768         | Graph, DS, FP16,  |       3.16          |
| 2.1-v           |    D910x1-MS2.1       |    LoRA      |      4x1                 |     768x768         | Graph, DS, FP16,  |       3.39          |
| 1.5           |    D910*x1-MS2.2      |    Vanilla   |      3x1             |     512x512         | Graph, DS, FP16,  |       9.12          |
| 1.5           |    D910*x1-MS2.2      |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |       10.72          |
| 1.5           |    D910*x1-MS2.2      |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |       2.33          |
| 2.0           |    D910*x1-MS2.2       |    Vanilla      |      3x1             |     512x512         | Graph, DS, FP16,  |         9.87        |
| 2.0           |    D910*x1-MS2.2       |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |            12.31     |
| 2.0           |    D910*x1-MS2.2       |    Dreambooth      |      1x1             |     512x512         | Graph, DS, FP16,  |        2.99         |
| 2.1-v           |    D910*x1-MS2.2       |    Vanilla      |      3x1             |     768x768         | Graph, DS, FP16,  |         5.18        |
| 2.1-v           |    D910*x1-MS2.2       |    LoRA      |      4x1                 |     768x768         | Graph, DS, FP16,  |         5.97        |
> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
>
> Acceleration: DS: data sink mode, FP16: float16 computation. Flash attention is not used in the test currently.
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
> Speed (ste/s): sampling speed measured in the number of sampling steps per second.
> FPS (img/s): image generation throughput measured in the number of image generated per second.

Note that the performance of SD2.1 should be similar to SD2.0 since they have the same network architecture.


<!--
Add a column for model/pipeline yaml config?
Mixed precision belongs to configuration
-->

## Image-to-Image

Coming soon
