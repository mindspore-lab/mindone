# ControlNet Training on 910B

## Environment

- Hardware: 910B
- MindSpore: 2.2 20231114

## Train a ControlNet from SD1.5

### 1. Model Weight Conversion

Once the [stable diffusion v1.5 model weights](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) (for mindspore)  are saved in `models`, you can run the following command to create a ControlNet init checkpoint.

```
python tools/sd_add_control.py
```

### 2. Data Preparation

We will use [Fill50k dataset](https://openi.pcl.ac.cn/attachments/5208caad-1727-46cc-b34e-add9afbd0557?type=1) to let the model learn to generate images following the edge control. Download it and put it under `datasets/` folder

For convenience, we take the first 1K samples as training set, which can done by keeping the first 1000 lines in `datasets/fill50k/prompt.json` and removing the rest.

If you want to use your own dataset, please make sure the structure as belows:
```text
dir
    ├── prompt.json
    ├── source
    └── target
```

`source` and `target` is the file folder with all images. The difference is the images under `target` folder are original image or called target image, and the images under `source` are the canny/segementation/other control images generated from original images. (eg.for Fill50k dataset `source/img0.png` is the canny image of `target/img0.png` )

```text
dir
├── img1.png
├── img2.png
├── img3.png
└── ...
```

`img_txt.csv` is the annotation file in the following format

```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```
### 3. Training


We will use the `scripts/run_train_cldm.sh` script for finetuning. Before running, please make sure the arguments `data_path` and `pretrained_model_path` are set to your own path, for example

```shell
data_path=/path_to_dataset
pretrained_model_path=/path_to_init_model
```

Then, check the training settings in `train_config`, some params default setting are as belows:

```text
train_batch_size: 2
optim: "adamw"
start_learning_rate: 5.e-4
```

For more settings, check the `model_config` file.

The default `sd_locked` is True, which means only update the parameters of ControlNet model, not the decoder part of Unet model.
The default `only_mid_control` is False, which means the data processed by ControlNet will go through the decoder part of Unet model.

For more explanation, please see [ControlNet official](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md#other-options)

Final, execute the script to launch finetuning

```
sh scripts/run_train_cldm.sh $CARD_ID
```

The resulting log will be saved in $output_dir as defined in the script, and the saved checkpoint will be saved in $output_path as defined in  `train_config` file.


### 4. Evaluation
To evalute the training result, please run the following script and indicate the path to the trained checkpoint.

```
sh scripts/run_infer_cldm.sh $CARD_ID $CHECKPOINT_PATH
```

And modify the control image path in the script.
