# Stable Diffusion Benchmark

## Text-to-Image

### Training

experiments are tested on ascend 910 with mindspore 2.1.0 graph mode
| sd model      |     method      | batch size |   resolution       |   flash attention   | jit_level  |fps (img/s)  |
|:----------:|:--------------:|:------------:|:------------------:|:----------------:|:----------------:|:----------:|
| 1.5           |    Vanilla   |      3x1             |     512x512         |  OFF  |     N/A |       5.53          |
| 1.5           |    Vanilla   |      3x8            |     512x512         |  OFF  |    N/A |     45.74          |
| 1.5           |    LoRA      |      4x1             |     512x512         |  OFF  |      N/A |   7.75          |
| 1.5           |    LoRA      |      4x8             |     512x512         |  OFF  |      N/A |  64.15          |
| 1.5           |    Dreambooth      |      1x1             |     512x512         |  OFF  |   N/A |   1.95            |
| 2.0           |    Vanilla      |      3x1             |     512x512         |  OFF  |     N/A |   6.17          |
| 2.0           |    Vanilla      |      3x8             |     512x512         |  OFF  |     N/A |   51.49          |
| 2.0           |    LoRA      |      4x1             |     512x512         |  OFF  |       N/A |   9.01       |
| 2.0           |    LoRA      |      4x8             |     512x512         |  OFF  |      N/A |    76.52       |
| 2.0           |    Dreambooth      |      1x1             |     512x512         |  OFF  |     N/A |  1.97            |
| 2.1-v           |    Vanilla      |      3x1             |     768x768         |   ON  |      N/A |   5.43          |
| 2.1-v           |    Vanilla      |      3x8             |     768x768         |   ON  |     N/A |    45.21          |
| 2.1-v           |    LoRA      |      4x1                 |     768x768         |   ON  |      N/A |   8.92          |
| 2.1-v           |    LoRA      |      4x8                 |     768x768         |   ON  |     N/A |    76.56          |

experiments are tested on ascend 910* with mindspore 2.3.0 graph mode
| sd model      |     method      | batch size |   resolution       |   flash attention   | jit_level  |fps (img/s)  |
|:----------:|:--------------:|:------------:|:------------------:|:----------------:|:----------------:|:----------:|
| 1.5           |       Vanilla   |      3x1             |     512x512         |  OFF  |     O2 |   11.86          |
| 1.5           |        Vanilla   |      3x8             |     512x512         |  OFF  |  O2 |  75.53          |
| 1.5           |       LoRA      |      4x1             |     512x512         |  OFF  |   O2 |  15.27          |
| 1.5           |        LoRA      |      4x8             |     512x512         |  OFF  |   O2 |  119.94          |
| 1.5           |       Dreambooth      |      1x1             |     512x512         |  OFF  |  O2 |    3.86          |
| 2.0           |        Vanilla      |      3x1             |     512x512         |  OFF  |     O2 |   12.75        |
| 2.0           |        Vanilla      |      3x8             |     512x512         |  OFF  |   O2 |    79.67        |
| 2.0           |        LoRA      |      4x1             |     512x512         |  OFF  |      O2 |     16.53     |
| 2.0           |        LoRA      |      4x8             |     512x512         |  OFF  |     O2 |      129.70     |
| 2.0           |        Dreambooth      |      1x1             |     512x512         |  OFF  |  O2 |   3.76         |
| 2.1-v           |        Vanilla      |      3x1             |     768x768         |   ON  |   O2 |   7.16        |
| 2.1-v           |        Vanilla      |      3x8             |     768x768         |   ON  |   O2 |    49.27        |
| 2.1-v           |        LoRA      |      4x1                 |     768x768         |   ON |   O2 |   9.51        |
| 2.1-v           |        LoRA      |      4x8                 |     768x768         |   ON  |   O2 |    71.51        |

experiments are tested on ascend 910* with mindspore 2.3.1 graph mode
| sd model      |     method      | batch size |   resolution       |   flash attention   | jit_level  |fps (img/s)  |
|:----------:|:--------------:|:------------:|:------------------:|:----------------:|:----------------:|:----------:|
| 1.5           |       Vanilla   |      3x1             |     512x512         |  OFF  |     O0 |   7.52          |
| 1.5           |        Vanilla   |      3x8             |     512x512         |  OFF  |  O0 |  34.43          |
| 1.5           |       LoRA      |      4x1             |     512x512         |  OFF  |   O0 |  9.86          |
| 1.5           |        LoRA      |      4x8             |     512x512         |  OFF  |   O0 |  71.54          |
| 1.5           |       Dreambooth      |      1x1             |     512x512         |  OFF  |  O0 |    1.35          |
| 1.5           |       TextualInversion      |      1x1            |     512x512         |  OFF  |  O0 |    1.72          |
| 2.0           |        Vanilla      |      3x1             |     512x512         |  OFF  |     O0 |   7.47        |
| 2.0           |        Vanilla      |      3x8             |     512x512         |  OFF  |   O0 |    38.73        |
| 2.0           |        LoRA      |      4x1             |     512x512         |  OFF  |      O0 |     9.98     |
| 2.0           |        LoRA      |      4x8             |     512x512         |  OFF  |     O0 |      73.72     |
| 2.0           |        Dreambooth      |      1x1             |     512x512         |  OFF  |  O0 |   1.19         |
| 2.0           |       TextualInversion      |      1x1            |     512x512         |  OFF  |  O0 |    1.23          |
| 2.1-v           |        Vanilla      |      3x1             |     768x768         |   ON  |   O0 |   6.46        |
| 2.1-v           |        Vanilla      |      3x8             |     768x768         |   ON  |   O0 |    43.06        |
| 2.1-v           |        LoRA      |      4x1                 |     768x768         |   ON |   O0 |   6.88        |
| 2.1-v           |        LoRA      |      4x8                 |     768x768         |   ON  |   O0 |    57.42        |

>fps: images per second during training. average training time (s/step) = batch_size / fps
>
>jie_level: control the compilation optimization level. N/A means the respective MindSpore version does not have the option.


<!--
TB tested:
| 1.5           |    ControlNet      |      4x1             |     512x512         |  OFF  |                 |
| 2.1-v           |    Dreambooth      |      1             |     768x768         |  OFF  |                 |
| 1.5           |    D910*-MS2.2      |    ControlNet      |      4x1             |     512x512         |  OFF  |                 |

Other Acceleration techniques:
Flash Attention,
-->


### Inference
experiments are tested on ascend 910 with mindspore 2.1.0 graph mode
| sd model     |  scheduler   | steps    |  resolution   |      batch size   |  jit_level  |  speed (step/s)     | fps (img/s)     |
|:---------------:|:------------:|:----------------:|:-------------:|:------------:|:----------------:|:----------:|:----------:|
| 1.5           |  DDIM       |   30       |    512x512         |       4     | N/A     |    3.11     |         0.37      |
| 2.0           |  DDIM       |   30       |    512x512         |       4     | N/A      |    3.58     |         0.42      |
| 2.1-v         |  DDIM       |   30       |    768x768         |       4     | N/A      |     1.14     |         0.14      |


experiments are tested on ascend 910* with mindspore 2.3.0 graph mode
| sd model     |  scheduler   | steps    |  resolution   |      batch size   |  jit_level  |  speed (step/s)     | fps (img/s)     |
|:---------------:|:------------:|:----------------:|:-------------:|:------------:|:----------------:|:----------:|:----------:|
| 1.5           |  DDIM       |   30       |    512x512         |       4     | O2      |       6.69     |         0.77      |
| 2.0           |  DDIM       |   30       |    512x512         |       4      | O2     |      8.30     |         0.91      |
| 2.1-v         |  DDIM       |   30       |    768x768         |       4      | O2     |      2.91     |         0.36      |

experiments are tested on ascend 910* with mindspore 2.3.1 graph mode
| sd model     |  scheduler   | steps    |  resolution   |      batch size   |  jit_level  |  speed (step/s)     | fps (img/s)     |
|:---------------:|:------------:|:----------------:|:-------------:|:------------:|:----------------:|:----------:|:----------:|
| 1.5           |  DDIM       |   30       |    512x512         |       4     | O0      |       5.72     |         0.45      |
| 2.0           |  DDIM       |   30       |    512x512         |       4      | O0     |      9.60     |         1.05      |
| 2.1-v         |  DDIM       |   30       |    768x768         |       4      | O0     |      4x1.68     |         0.49      |

> Speed (step/s): sampling speed measured in the number of sampling steps per second.
>
> fps (img/s): image generation throughput measured in the number of image generated per second.
>
>jie_level: control the compilation optimization level. N/A means the respective MindSpore version does not have the option.


<!--
Add a column for model/pipeline yaml config?
Mixed precision belongs to configuration
-->
