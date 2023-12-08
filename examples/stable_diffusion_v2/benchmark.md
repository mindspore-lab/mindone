# Benchmark

## Text-to-Image

### Inference

| SD Model      |     Context |  Scheduler   | Steps              |  Resolution   |      Batch Size     |  FPS (img/s)     | 
|---------------|:-----------|:------------:|:------------------:|:----------------:|:----------------:|:----------------:|
| 1.5           |     D910x1-MS2.0    |  DPM++       |   25               |    512x512         |       4          |      |
| 2.0           |     D910x1-MS2.0    |  DPM++       |   25               |    512x512         |       4          |      |
| 2.1-v         |     D910x1-MS2.0    |  DPM++       |   25               |    768x768         |       4          |      |
| 1.5           |     D910*x1-MS2.2   |  DPM++       |   25               |    512x512         |       4          |      |    
| 2.0           |     D910*x1-MS2.2   |  DPM++       |   25               |    512x512         |       4          |      |
| 2.1-v         |     D910*x1-MS2.2   |  DPM++       |   25               |    768x768         |       4          |      |
> Context: {Ascend chip}-{number of NPUs}-{mindspore version}

Note that the performance of SD2.1 should be similar to SD2.0 since they have the same network architecture. 


<!--
Add a column for model/pipeline yaml config?
Mixed precision belongs to configuration
-->

### Training

| SD Model      |   Context      |  Method      | Batch Size x Grad. Accu. |   Resolution       |   Acceleration   |   FPS (img/s)  | 
|---------------|---------------|--------------|:-------------------:|:------------------:|:----------------:|:----------------:|
| 1.5           |    D910x1-MS2.0      |    Vanilla   |      3x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 1.5           |    D910x1-MS2.0      |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 1.5           |    D910x1-MS2.0      |    Dreambooth      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 1.5           |    D910x1-MS2.0      |    ControlNet      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 2.0           |    D910x1-MS2.0       |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 2.0           |    D910x1-MS2.0       |    Dreambooth      |      4x1             |     768x768         | Graph, DS, FP16,  |                 |     
| 2.1-v           |    D910x1-MS2.0       |    Vanilla      |      4x1             |     768x768         | Graph, DS, FP16,  |                 |     
| 2.1-v           |    D910x1-MS2.0       |    LoRA      |      4x1                 |     768x768         | Graph, DS, FP16,  |                 |     
| 2.1-v           |    D910x1-MS2.0       |    Dreambooth      |      4x1             |     768x768         | Graph, DS, FP16,  |                 |     
| 1.5           |    D910*x1-MS2.2      |    Vanilla   |      3x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 1.5           |    D910*x1-MS2.2      |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 1.5           |    D910*x1-MS2.2      |    Dreambooth      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 1.5           |    D910*x1-MS2.2      |    ControlNet      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 2.0           |    D910*x1-MS2.2       |    LoRA      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 2.0           |    D910*x1-MS2.2       |    Dreambooth      |      4x1             |     512x512         | Graph, DS, FP16,  |                 |     
| 2.1-v           |    D910*x1-MS2.0       |    Vanilla      |      4x1             |     768x768         | Graph, DS, FP16,  |                 |     
| 2.1-v           |    D910*x1-MS2.0       |    LoRA      |      4x1                 |     768x768         | Graph, DS, FP16,  |                 |     
| 2.1-v           |    D910*x1-MS2.0       |    Dreambooth      |      4x1             |     768x768         | Graph, DS, FP16,  |                 | 

> DS: data sink mode, FP16: float16 computation.
>
> FPS: images per second during training. average training time (s/step) = batch_size / FPS  

Note that the performance of SD2.1 should be similar to SD2.0 since they have the same network architecture. 

Note that SD1.x and SD2.x share the same UNet architecture, thus their performance on vanilla training are similar.

<!--
Other Acceleration techniques: 
Flash Attention, 
-->

## Image-to-Image

Coming soon
