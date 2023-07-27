# Dreambooth for Stable Diffusion Finetuning
[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)

## 1. Introduction

DreamBooth is a parameter-efficient method used to fine-tune existing text-to-image models, developed by researchers from Google Research and Boston University in 2022. It allows the model to generate contextualized images of the subject, e.g., a cute puppy of yours, in different scenes, poses, and views. DreamBooth can be used to fine-tune models such as Stable Diffusion2.

The following picture gives a high-level overview of the DreamBooth Method.

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/images/dreambooth_diagram.png" width=850 />
</p>
<p align="center">
  <em> Figure 1. The Diagram of DreamBooth. [<a href="#references">1</a>] </em>
</p>


DreamBooth only requires a pretrained Text-to-Image model and a few (3-5) subject's images along with its name as a Unique Identifier. Unique Identifier prevents the Text-to-Image model from **language drift** [<a href="#references">2</a>]. Language Drift is a phenonomon that the model tends to associate the class name (e.g., "dog") with the specific instance (e.g., your dog). Because during finetuning, the model only sees the prompt "dog" and your dog's pictures, so that it gradually forgets other dogs' look. Therefore, we need a Unique Identifier ("sks") to differentiate your a dog ("sks dog") and a general dog ("dog").

To do this, authors came up with a class-specific prior preservation loss:

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/images/dreambooth_loss.PNG" width=650 />
</p>
<p align="center">
  <em> Equation 1. The class-specific prior preservation loss. [<a href="#references">1</a>] </em>
</p>

$x$ is the image of your subject ("sks dog"), and $x_{pr}$ is the image from the same class as your subject ("dog"). $\hat{x}_{\theta}$ is the pretrained Text-to-Image model, which takes a nosiy image input and the condiction generated from the text encoder, and outputs a de-noised image. The second term of the equation above works as a prior preservation term that prevents the model from forgetting the looks of other dogs.


**Notes**:
- Unlike LoRA, Dreambooth is a method that updates all the weights of the text-to-image model. If needed, the text encoder can also be updated. We find that finetuning text encoder and the text-to-image model yields better performance than finetuning the text-to-image model alone.

## 2. Get Started

**MindONE** supports DreamBooth finetuning for Stable Diffusion models based on MindSpore and Ascend platforms.

### 2.1 Preparation

#### 2.1.1 Dependency

Please make sure the following frameworks are installed.

- mindspore >= 1.9  [[install](https://www.mindspore.cn/install)] (2.0 is recommended for the best performance.)
- python >= 3.7
- openmpi 4.0.3 (for distributed training/evaluation)  [[install](https://www.open-mpi.org/software/ompi/v4.0/)]

Install the dependent packages by running:
```shell
pip install -r requirements.txt
```

#### 2.1.2 Pretrained Models

Please download the pretrained [SD2.0-base checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) and put it under `stable_diffusion_v2/models` folder.


#### 2.1.3 Finetuning Dataset Preparation

The finetuning dataset should contain 3-5 images from the same subject under the same folder.

```text
dir
├── img1.jpg
├── img2.jpg
├── img3.jpg
├── img4.jpg
└── img5.jpg
```

In [Google/DreamBooth](https://github.com/google/dreambooth), there are many images from different classes. In this tutorial, we will use the [five images](https://github.com/google/dreambooth/tree/main/dataset/dog) from this dog. They are shown below:

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/images/dreambooth_finetune_images.png" width=650 />
</p>
<p align="center">
  <em> Figure 2. The five images from the subject dog for finetuning. </em>
</p>

### 2.2 Finetuning

#### 2.2.1 Environmental Variables

Before starting to run the finetuning program, some environmental variables are recommended to be set like:

```bash
# Ascend settings
export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# Choose stable diffusion version
export SD_VERSION="2.0"

# standalone training settings
device_id=0
export RANK_SIZE=1
export DEVICE_ID=$device_id
```

#### 2.2.2 Experiment-Related Variables

Next, we define some experiment-related variables that may vary for different users:

```bash
# checkpoints will be saved in ${output_path}/${task_name}
output_path=output/
task_name=txt2img
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}; \

# define the instance data-dir/prompt and class data-dir/prompt
instance_data_dir=dreambooth/dataset/dog
instance_prompt="a photo of sks dog"
class_data_dir=temp_class_images/dog
class_prompt="a photo of a dog"

# the weight file of the pretrained stable diffusion
pretrained_model_path=models/
pretrained_model_file=sd_v2_base-57526ee4.ckpt

# the training configuration file. The arguments passed via command
# line will overwrite what is in this configuration file
train_config_file=configs/train_dreambooth_sd_v2.json

# On Ascend 910 (30GB), when image size is (512, 512), train batch size
# is recommended to be 1. On other hardware configuration, one can alter
# them accordingly.
image_size=512
train_batch_size=1
```


#### 2.2.3 Training Command for DreamBooth

In `scripts/run_train_dreambooth_sd_v2.sh`, the training command is like this:
```bash
python train_dreambooth.py \
    --mode=0 \
    --use_parallel=False \
    --instance_data_dir=$instance_data_dir \
    --instance_prompt="$instance_prompt"  \
    --class_data_dir=$class_data_dir \
    --class_prompt="$class_prompt" \
    --train_config=$train_config_file \
    --output_path=$output_path/$task_name \
    --pretrained_model_path=$pretrained_model_path \
    --pretrained_model_file=$pretrained_model_file \
    --image_size=$image_size \
    --train_batch_size=$train_batch_size \
    --epochs=8 \
    --start_learning_rate=2e-6 \
    --train_text_encoder=True \
    # --train_text_encoder=False \
```

Using the command above, we will start standalone training (`use_parallel=False`) with a constant learning rate (`2e-6`) for 8 epochs (1600 steps).

The `num_class_images` in the arguments is 200 by default. It is the number of class images for prior preservation loss. If there are not enough images already present in `class_data_dir`, additional images will be sampled with `class_prompt`. We also resample the instance images by `train_data_repeats` times so that the numbers of class images and instance images are the same.

The finetuning process takes about 30 minutes.

**Notes**:
> 1. Set `train_text_encoder` to `True` allows to finetune the stable diffusion model along with the CLIP text encoder. We recommend to set it to True, since it yields better performance than `train_text_encoder=False`.
> 2. If `train_text_encoder` is set to `False` which saves some memory, we recommend you to change the `epochs` to 20 to achieve better performance.

#### 2.2.4 Training Command for Vanilla Finetuning

Vanilla Finetuning is to finetune the network with the five dogs images directly, without applying the prior perservation loss.

To run Vanilla Finetuning, you can set the Environmental Variables as <a href="#221-environmental-variables">2.2.1</a> and the Experiment-Related Variables as <a href="#222-experiment-related-variables">2.2.2</a> (ignore `class_data_dir` and `class_prompt`). The training command for vanilla finetuning is:

```bash
python train_dreambooth.py \
    --mode=0 \
    --use_parallel=False \
    --instance_data_dir=$instance_data_dir \
    --instance_prompt="$instance_prompt"  \
    --train_config=$train_config_file \
    --output_path="output_vanilla_ft/txt2img" \
    --pretrained_model_path=$pretrained_model_path \
    --pretrained_model_file=$pretrained_model_file \
    --image_size=$image_size \
    --train_batch_size=$train_batch_size \
    --epochs=8 \
    --start_learning_rate=2e-6 \
    --train_text_encoder=True \
    --with_prior_preservation=False \
```

### 2.3 Inference

The inference command generates images on a given prompt (e.g., "a sks dog on the hill") and save them to a given output directory. An example command is as follows:

```bash
python text_to_image.py \
    --prompt "a sks dog on the hill" \
    --output_path vis/output/dir \
    --config configs/train_dreambooth_sd_v2.yaml \
    --ckpt_path path/to/checkpoint/file
```

You can also change the `prompt` to generate variant images with the subject. `path/to/checkpoint/file` specifies the checkpoint path after finetuing.

Here are some examples of generated images with the DreamBooth model using this prompt:

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/images/dreambooth_grid_epoch_8.png" width=650 />
</p>
<p align="center">
  <em> Figure 3. The generated images of "a sks dog on the hill" with the DreamBooth model. </em>
</p>

Some generated images with the vanilla finetuned model using the same prompt are shown below:

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/images/vanilla_ft_grid_epoch_8.png" width=650 />
</p>
<p align="center">
  <em> Figure 4. The generated images of "a sks dog on the hill" with the vanilla finetuned model. </em>
</p>

### 2.4 Evaluation

For evaluation, we consider two metrics:

- CLIP-T score: the cosine similarity between the text prompt and the image CLIP embedding.
- CLIP-I score: the cosine similarity between the CLIP embeddings of the real and the generated images.

By default, we use the pretrained `clip_vit_base_patch16` model to extract the image/text embeddings and `bpe_simple_vocab_16e6` as the text tokenizer. To download the two files, please refer to the guideline [Clip Score Evaluation](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_v2/eval/README.md#clip-score).


#### 2.4.1 CLIP-T score

The command to evaluate the CLIP-T score is shown below:

```shell
python eval/eval_clip_score.py  \
    --load_checkpoint <path-to-clip-model>  \
    --image_path_or_dir <path-to-image>  \
    --prompt_or_path <string/path-to-txt>
```

#### 2.4.2 CLIP-I score

The command to evaluate the CLIP-I score is shown below:

```shell
python eval/eval_clip_i_score.py  \
    --load_checkpoint <path-to-clip-model>  \
    --gen_image_path_or_dir <path-to-generated-image>  \
    --real_image_path_or_dir <path-to-real-image>
```

# References

[1] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman:
DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CoRR abs/2208.12242 (2022)

[2]: 	Jason Lee, Kyunghyun Cho, Douwe Kiela:
Countering Language Drift via Visual Grounding. CoRR abs/1909.04499 (2019)
