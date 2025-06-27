# MMaDA: Multimodal Large Diffusion Language Models
## 🌌 Introduction
This is a MindSpore implementation of [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809). MMaDA represents a unified multimodal diffusion model that can generate text, images, and multimodal reasoning. The following demo shows how MMaDA generates text, images, and multimodal reasoning.

<div align="center" style="width: 600px; margin: auto;">
  <img src="https://github.com/Gen-Verse/MMaDA/blob/main/assets/showcase0.8.gif?raw=true" alt="MMaDA decoding demo" width="550" />
  <p style="font-style: italic; font-size: 14px; color: #555; margin-top: 6px;">
    MMaDA's decoding demo. This video showcases how a diffusion foundation model generates text and image.<br>
    Image source: https://github.com/Gen-Verse/MMaDA
  </p>
</div>

There are three key constributions of this model:

1. MMaDA is a unified diffusion model for multiple modalities and tasks.
2. MMaDA incorporates a mixed long chain-of-thought (**CoT**) fine-tuning methodology.
3. MMaDA implements **UniGRPO**, a novel policy-gradient-based reinforcement learning algorithm specifically optimized for diffusion foundation models.


## 📑 Development Plan

Here is the development plan of the project:

- MMaDA (8B) Inference:
    - [x] Text generation
    - [x] Multimodal reasoning
    - [x] Text-to-image generation
    - [ ] Gradio Demo
- MMaDA (8B) Training:
    - [x] Pre-training
    - [x] Fine-tuning




## 📦 Requirements


<div align="center">

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.6.0   |  24.1.RC3     | 7.6.0.1.220 |  8.0.RC3.beta1     |

</div>

1. Install
   [CANN 8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1)
   and MindSpore according to the [official instructions](https://www.mindspore.cn/install).
2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```
3. Install mindone
    ```
    cd mindone
    pip install -e .
    ```
    Try `python -c "import mindone"`. If no error occurs, the installation is successful.

## 🚀 Quick Start

### Checkpoints

You can download `MMaDA-8B-Base` at [Huggingface](https://huggingface.co/Gen-Verse/MMaDA-8B-Base) using the following coomand:
```bash
huggingface-cli download --resume-download Gen-Verse/MMaDA-8B-Base
```

And download `MMaDA-8B-MixCoT` from [Huggingface](https://huggingface.co/Gen-Verse/MMaDA-8B-MixCoT) like this:

```bash
huggingface-cli download --resume-download Gen-Verse/MMaDA-8B-MixCoT
```

`MMaDA-8B-Max` is comining soon. See latest updates from [HERE](https://github.com/Gen-Verse/MMaDA/blob/main/README.md#-mmada-series-overview).

### Inference Files Preparation

Please download the image files and text files for multimodal generation from this [URL](https://huggingface.co/datasets/ddengwtomin/mmada-repository/tree/main). You can also download them using the following command:
```bash
cd examples/mmada
huggingface-cli download --resume-download ddengwtomin/mmada-repository --local-dir ./ --exclude "README.md" ".gitattributes" "train_datasets/*" --repo-type dataset
```

### 1. Text Generation

For text generation, please run:
```bash
python generate.py
```

### 2. MultiModal Generation

For multiModal generation, please run:
```
python3 inference_mmu.py config=configs/mmada_demo.yaml mmu_image_root=./mmu_validation question='Please describe this image in detail.'
```

The outputs are stored locally.

### 3. Text-to-Image Genertion
For text-to-image generation, please run:
```
python3 inference_t2i.py config=configs/mmada_demo.yaml batch_size=1 validation_prompts_file=validation_prompts/text2image_prompts.txt guidance_scale=3.5 generation_timesteps=15
mode='t2i'
```
The outputs are stored locally.

## 🔧 Training


### Training Datasets

Please refer to the Official PyTorch Repository [MmaDA](https://github.com/Gen-Verse/MMaDA) for the full information about the training datasets. You can download the relevant training datasets following the instructions from this [document](./docs/datasets.md).

For demonstration purpose, we provide a toy example of training datasets in [hugginfgace](https://huggingface.co/datasets/ddengwtomin/mmada-repository/tree/main/train_datasets). You can download this toy dataset using:
```bash
cd examples/mmada
huggingface-cli download --resume-download ddengwtomin/mmada-repository --local-dir ./ --include "train_datasets/*" --repo-type dataset
```

Afterwards, three datasets (t2i, text, and multimodal) will be downloaded in `./train_datasets`.


### Pretraining Experiment

After the training datasets are downloaded, please edit `configs/mmada_pretraining_stage1_llada_instruct.yaml` and revise the following dataset paths accordingly:
```yaml
train_t2i_shards_path_or_url: "path/to/your/dataset"
train_mmu_shards_path_or_url: "path/to/your/dataset"
train_lm_shards_path_or_url: "path/to/your/dataset"
```

Then you can start the standalone training experiment with the following command:
```bash
python training/train_mmada.py config=configs/mmada_pretraining_stage1_llada_instruct.yaml
```

The experiment logs and checkpoints will be saved under `./mmada-training-stage1-llada-instruct`, as defined by the `experiment.output_dir` in the configuration file.

We recommend you to start a ZERO2 parallel training task with `scripts/pretrain_stage1_parallel.sh`:
```bash
export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=./parallel_logs \
python training/train_mmada.py config=configs/mmada_pretraining_stage1_llada_instruct.yaml
```

### Finetuning Experiment

We provide a configuration file `configs/mmada_finetune_artwork.yaml` for finetuning experiment. Please start the finetuning experiment with `scripts/finetune_artwork.sh`:
```bash
export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=./parallel_logs \
python training/train_mmada_stage2.py config=configs/mmada_finetune_artwork.yaml
```


## 🤝 Acknowledgments

We would like to thank the contributors to the [MMaDA](https://github.com/Gen-Verse/MMaDA), [LLaDA](https://github.com/ML-GSAI/LLaDA), [maskgit](https://github.com/google-research/maskgit), [transformers](https://github.com/huggingface/transformers), [transformers](https://github.com/huggingface/transformers), and [webdataset](https://github.com/webdataset/webdataset)repositories, for their open research and exploration.
