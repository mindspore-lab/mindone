# Stable Diffusion Benchmark

## Text-to-Image

### Training

experiments are tested on ascend 910 with mindspore 2.1.0 graph mode
|  model name      |     method      | cards | batch size |   resolution       |   flash attn   | jit level  |img/s  |
|:----------:|:--------------:|:--------:|:----:       |:------------------:|:----------------:|:----------------:|:----------:|
| sd 1.5           |    Vanilla   |   1   | 3             |     512x512         |  OFF  |     N/A |       5.53          |
| sd 1.5           |    Vanilla   |   8   | 3            |     512x512         |  OFF  |    N/A |     45.74          |
| sd 1.5           |    LoRA      |   1   | 4             |     512x512         |  OFF  |      N/A |   7.75          |
| sd 1.5           |    LoRA      |   8   | 4             |     512x512         |  OFF  |      N/A |  64.15          |
| sd 1.5           |    Dreambooth|   1   | 1             |     512x512         |  OFF  |   N/A |   1.95            |
| sd 2.0           |    Vanilla   |   1   | 3             |     512x512         |  OFF  |     N/A |   6.17          |
| sd 2.0           |    Vanilla   |   8   | 3             |     512x512         |  OFF  |     N/A |   51.49          |
| sd 2.0           |    LoRA      |   1   | 4             |     512x512         |  OFF  |       N/A |   9.01       |
| sd 2.0           |    LoRA      |   8   | 4             |     512x512         |  OFF  |      N/A |    76.52       |
| sd 2.0           |    Dreambooth|   1   | 1             |     512x512         |  OFF  |     N/A |  1.97            |
| sd 2.1-v           |    Vanilla |   1   | 3             |     768x768         |   ON  |      N/A |   5.43          |
| sd 2.1-v           |    Vanilla |   8   | 3             |     768x768         |   ON  |     N/A |    45.21          |
| sd 2.1-v           |    LoRA    |   1   | 4                 |     768x768         |   ON  |      N/A |   8.92          |
| sd 2.1-v           |    LoRA    |   8   | 4                 |     768x768         |   ON  |     N/A |    76.56          |

experiments are tested on ascend 910* with mindspore 2.3.0 graph mode
| model name     |     method      |    cards | batch size  |   resolution       |   flash attn   | jit level  | img/s  |
|:----------:|:--------------:|         :----:|:----:                 |:------------------:|:----------------:|:----------------:|:----------:|
| sd 1.5           |       Vanilla    |   1   | 3                 |     512x512         |  OFF  |     O2 |   11.86          |
| sd 1.5           |        Vanilla   |   8   | 3                 |     512x512         |  OFF  |  O2 |  75.53          |
| sd 1.5           |       LoRA       |   1   | 4                 |     512x512         |  OFF  |   O2 |  15.27          |
| sd 1.5           |        LoRA      |   8   | 4                 |     512x512         |  OFF  |   O2 |  119.94          |
| sd 1.5           |       Dreambooth |   1   | 1                 |     512x512         |  OFF  |  O2 |    3.86          |
| sd 2.0           |        Vanilla   |   1   | 3                 |     512x512         |  OFF  |     O2 |   12.75        |
| sd 2.0           |        Vanilla   |   8   | 3                 |     512x512         |  OFF  |   O2 |    79.67        |
| sd 2.0           |        LoRA      |   1   | 4                 |     512x512         |  OFF  |      O2 |     16.53     |
| sd 2.0           |        LoRA      |   8   | 4                 |     512x512         |  OFF  |     O2 |      129.70     |
| sd 2.0           |        Dreambooth|   1   | 1                 |     512x512         |  OFF  |  O2 |   3.76         |
| sd 2.1-v           |        Vanilla |   1   | 3                 |     768x768         |   ON  |   O2 |   7.16        |
| sd 2.1-v           |        Vanilla |   8   | 3                 |     768x768         |   ON  |   O2 |    49.27        |
| sd 2.1-v           |        LoRA    |   1   | 4                     |     768x768         |   ON |   O2 |   9.51        |
| sd 2.1-v           |        LoRA    |   8   | 4                     |     768x768         |   ON  |   O2 |    71.51        |

experiments are tested on ascend 910* with mindspore 2.3.1 graph mode
| model name      |     method      |   cards | batch size    |   resolution       |   flash attn   | jit level  | img/s  |
|:----------:|:--------------:| :--:|        :------------:|:------------------:|:----------------:|:----------------:|:----------:|
| sd 1.5           |       Vanilla    |   1 |  3             |     512x512         |  OFF  |     O0 |   7.52          |
| sd 1.5           |        Vanilla   |   8 |  3             |     512x512         |  OFF  |  O0 |  34.43          |
| sd 1.5           |       LoRA       |   1 |  4             |     512x512         |  OFF  |   O0 |  9.86          |
| sd 1.5           |        LoRA      |   8 |  4             |     512x512         |  OFF  |   O0 |  71.54          |
| sd 1.5           |       Dreambooth  |  1 |  1             |     512x512         |  OFF  |  O0 |    1.35          |
| sd 1.5           |  TextualInversion |  1 |  1            |     512x512         |  OFF  |  O0 |    1.72          |
| sd 2.0           |        Vanilla    |  1 |  3             |     512x512         |  OFF  |     O0 |   7.47        |
| sd 2.0           |        Vanilla    |  8 |  3             |     512x512         |  OFF  |   O0 |    38.73        |
| sd 2.0           |        LoRA      |   1 |  4             |     512x512         |  OFF  |      O0 |     9.98     |
| sd 2.0           |        LoRA      |   8 |  4             |     512x512         |  OFF  |     O0 |      73.72     |
| sd 2.0           |        Dreambooth|   1 |  1             |     512x512         |  OFF  |  O0 |   1.19         |
| sd 2.0           |  TextualInversion|   1 |  1            |     512x512         |  OFF  |  O0 |    1.23          |
| sd 2.1-v         |        Vanilla   |   1 |  3             |     768x768         |   ON  |   O0 |   6.46        |
| sd 2.1-v         |        Vanilla   |   8 |  3             |     768x768         |   ON  |   O0 |    43.06        |
| sd 2.1-v         |        LoRA      |   1 |  4                 |     768x768         |   ON |   O0 |   6.88        |
| sd 2.1-v         |        LoRA      |   8 |  4                 |     768x768         |   ON  |   O0 |    57.42        |

>fps (img/s) : images per second during training. average training time (s/step) = batch_size / fps
>
>jit level: control the compilation optimization level. N/A means the respective MindSpore version does not have the option.


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
| model name       |  batch size   |  resolution   |  scheduler  |  steps    |  jit level  | step/s     | img/s     |
|:---------------: |:------------: |:-------------:|:-----------:|:---------:|:----------------:|:----------:|:----------:|
| sd 1.5           |       4       |    512x512    |  DDIM       |   30      | N/A     |    3.11     |         0.37      |
| sd 2.0           |       4       |    512x512    |  DDIM       |   30      | N/A      |    3.58     |         0.42      |
| sd 2.1-v         |       4       |    768x768    |  DDIM       |   30      | N/A      |     1.14     |         0.14      |


experiments are tested on ascend 910* with mindspore 2.3.0 graph mode
| model name     |   batch size |  resolution   |  scheduler   | steps       |  jit level  | step/s     | img/s     |
|:--------------:|:------------:|:-------------:| :------------:|:----------:|:----------------:|:----------:|:----------:|
| sd 1.5         |       4      |    512x512    |    DDIM       |   30       | O2      |       6.69     |         0.77      |
| sd 2.0         |       4      |    512x512    |    DDIM       |   30       | O2     |      8.30     |         0.91      |
| sd 2.1-v       |       4      |    768x768    |    DDIM       |   30       | O2     |      2.91     |         0.36      |

experiments are tested on ascend 910* with mindspore 2.3.1 graph mode
| model name       |      batch size   |  resolution   |  scheduler  | steps    |  jit level  |  step/s     | img/s     |
|:---------------: |:------------:     |:-------------:|:-----------:|:--------:|:----------------:|:----------:|:----------:|
| sd 1.5           |       4           |    512x512    |  DDIM       |   30     | O0      |       5.72     |         0.45      |
| sd 2.0           |       4           |    512x512    |  DDIM       |   30     | O0     |      9.60     |         1.05      |
| sd 2.1-v         |       4           |    768x768    |  DDIM       |   30     | O0     |      4x1.68     |         0.49      |

> Speed (step/s): sampling speed measured in the number of sampling steps per second.
>
> fps (img/s): image generation throughput measured in the number of image generated per second.
>
>jit level: control the compilation optimization level. N/A means the respective MindSpore version does not have the option.


<!--
Add a column for model/pipeline yaml config?
Mixed precision belongs to configuration
-->
