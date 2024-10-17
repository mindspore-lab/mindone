# Installation

## Dependency

- AI framework and platform

| platform | mindspore      | ascend driver | firmware    | cann toolkit/kernel |
| -------- | -------------- | ------------- | ----------- | ------------------- |
| 910*     | 2.2.10～2.2.12 | 23.0.3        | 7.1.0.5.220 | 7.0.0 beta1         |

> Notes: All the features should work on MindSpore 2.2.10～2.2.12 on Ascend 910*. The finetune methods (vallina/lora/dreambooth) are adapted to MindSpore 2.3.0/2.3.1 on the [branch v0.2.0](https://github.com/mindspore-lab/mindone/tree/v0.2.0) and will be merged into the master branch later. In the future MindSpore version, `mindone.diffusers` are recommended for SDXL training and inference.

- openmpi 4.0.3 (for distributed mode)

To install the dependency, please run

```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instructions](https://www.mindspore.cn/install) to select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.
