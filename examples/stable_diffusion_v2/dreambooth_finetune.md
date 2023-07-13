# Dreambooth for Stable Diffusion Finetuning
[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)

## Introduction

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

## Get Started

**MindONE** supports DreamBooth finetuning for Stable Diffusion models based on MindSpore and Ascend platforms.

### Preparation

#### Dependency

Please make sure the following frameworks are installed.

- mindspore >= 1.9  [[install](https://www.mindspore.cn/install)] (2.0 is recommended for the best performance.)
- python >= 3.7
- openmpi 4.0.3 (for distributed training/evaluation)  [[install](https://www.open-mpi.org/software/ompi/v4.0/)]

Install the dependent packages by running:
```shell
pip install -r requirements.txt
```

#### Pretrained Models

Please download the pretrained [SD2.0-base checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) and put it under `stable_diffusion_v2/models` folder.


#### Finetuning Dataset Preparation

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

### Finetuning

**Environmental Variables**

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

**Experiment-Related Variables**

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


**Training Command**

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

**Notes**:
> 1. Set `train_text_encoder` to `True` allows to finetune the stable diffusion model along with the CLIP text encoder. We recommend to set it to True, since it yields better performance with less updates than `train_text_encoder=False`. If `train_text_encoder=False`, we recommend you to change the `epochs` to 20 to achieve better performance.

The finetuning process takes about 30 minutes.

**Testing Command**

The testing command is to run inference on the `$instance_prompt` and visualize the generated images. An example testing command is as follows:
```bash
python text_to_image.py \
    --prompt "a sks dog on the hill" \
    --output_path vis/output/dir \
    --config configs/train_dreambooth_sd_v2.yaml \
    --ckpt_path path/to/checkpoint/file
```

You can also change the `prompt` to generate variant images with the subject.

We passed the 8th-epoch checkpoint path to the testing command. Here are some examples of generated images using the prompt "a sks dog on the hill":


<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/images/dreambooth_grid_epoch_8.png" width=650 />
</p>
<p align="center">
  <em> Figure 3. The generated images of the given prompt. </em>
</p>


# References

[1] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman:
DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CoRR abs/2208.12242 (2022)

[2]: 	Jason Lee, Kyunghyun Cho, Douwe Kiela:
Countering Language Drift via Visual Grounding. CoRR abs/1909.04499 (2019)
