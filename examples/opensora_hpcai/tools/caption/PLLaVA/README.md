# PLLaVA based on MindSpore

MindSpore implementation of
[PLLaVA : Parameter-free LLaVA Extension from Images to Videos for Video Dense Captioning
](https://arxiv.org/abs/2404.16994).

## Dependencies

- CANN: 8.0.RC3.beta1 or later
- Python: 3.9 or later
- Mindspore: 2.4.1

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

| Video ID | Sample Frame | Caption |
|----------|--------------|---------|
| <video src="https://github.com/user-attachments/assets/e79c8b19-b5f6-4391-8bf4-4921e2fede15" /> | <img width="960" alt="2" src="https://github.com/user-attachments/assets/19615fcd-b0e9-431a-b882-fea75b43d84e" /> | The image shows a collection of cake pans inside an oven. Each pan has a different color of frosting, indicating that they are being used to bake cakes with various flavors or colors. The oven appears to be a professional-grade model, suitable for baking large quantities of cakes at once. The pans are arranged on a rack, which is designed to allow for even heat distribution and to prevent the cakes from sticking to the bottom of the oven. |


## Benchmark

### Inference

To test the benchmark, you may use the video `-0og5HrzhpY_0.mp4` under `./examples`
and run the following command:
```bash
python pllavarun.py --video ./example/-0og5HrzhpY_0.mp4 --benchmark
```

|         Model         | Context       | Batch Size | Throughput (tokens/second) |
|-----------------------|---------------|------------|----------------------------|
| pllava-7b| D910*x1-MS2.4 |    1       | 8.89                       |

> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.\
> Throughput (tokens/second): number of generated tokens per second.\
> We use the second round of inference as the benchmark result.
