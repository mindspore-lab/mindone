# Kohya sd-scripts on MindSpore

Here we provide a MindSpore implementation of  [Kohya's Stable Diffusion trainers](https://github.com/kohya-ss/sd-scripts).

Currently we support

* SDXL LoRA  training
* SDXL Inference

## Installing the dependencies

The scripts work on Acend 910* with [CANN 8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) and [MindSpore 2.3.0 ](https://www.mindspore.cn/versions#2.3.0). Check your versions by running the following commands. The default installation path of CANN is usually  `/usr/local/Ascend/ascend-toolkit` unless you specify a custom one. 

```bash
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg  
# see a version number as [7.3.0.1.231:8.0.RC2]

python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
# MindSpore version: 2.3.0
```

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/mindspore-lab/mindone
cd mindone
pip install .
```

Enter in the example folder `examples/kohya_sd_scripts`.

```bash
cd examples/kohya_sd_scripts
```



## SDXL LoRA

### Prepare learning data

Prepare image files for learning data in any folder (multiple folders are also acceptable). Supported formats are `.png`, `.jpg`, `.jpeg`, `.webp`, and `.bmp`.  

You can specify the learning data in several ways, depending on the number of learning data, learning target, whether captions (image descriptions) can be prepared, etc. Kohya LoRA script supports three methods as follows. Choose a method and config the datasets in a `.toml` file, which can be passed with the `--dataset_config` option to the training script.

| Learning target or method | Script                  | DB / class+identifier | DB / Caption | fine-tuning |
| ------------------------- | ----------------------- | --------------------- | ------------ | ----------- |
| LoRA                      | `sdxl_train_network.py` | o                     | o            | o           |

Here we take Pokemon dataset as examples to explain the three learning data patterns.  Please refer to [Config Readme](https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-en.md) and  [Common Learning Guide](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_README-en.md) from kohya's official repo for more details.



#### 1. DreamBooth, class+identifier method (regularization images can be used)

Learn by associating the learning target with a specific word (identifier). There is no need to prepare captions, each image is learned as if it was learned with the caption `class identifier`.  For example, img711 to img715 in Pokemon training set are butterflies, we can have the images in `data_path/butterfly` folder and train with class identifier `Pokemon butterfly`. If using regualtion images please prepare them in the same way.

```
data_path/butterfly
├── img711.png
├── img712.png
├── img713.png
├── img714.png
└── img715.png
```

Create a text file and set the extension to `.toml` and write as follow.

```toml
[general]
enable_bucket = false                        # Whether to use Aspect Ratio Bucketing

[[datasets]]
resolution = 1024                            # Training resolution
batch_size = 1                               # Batch size

  [[datasets.subsets]]
  random_crop = true												 # enable it when enable_bucket=false
  image_dir = 'data_path/butterfly'          # Specify the folder containing the training images
  class_tokens = 'Pokemon butterfly'         # Caption file extension; change this if using .txt
  num_repeats = 10                          `# Number of repetitions for training images

  # Write the following only when using regularization images. Remove it if not using them.
  [[datasets.subsets]]
  is_reg = true
  random_crop = true												 # enable it when enable_bucket=false
  image_dir = 'path_to_regulation_dataset'   # Specify the folder containing the regularization images
  class_tokens = 'butterfly'                 # Specify the class
  num_repeats = 1                            # Number of repetitions for regularization images; 1 is usually sufficient
```



#### 2. DreamBooth, caption method (regularization images can be used)

Prepare a text file with captions recorded for each image and learn. For example, when learning a specific character, by describing the details of the image in the caption, the character and other elements are separated, and the model can be expected to learn the character more precisely.

We still have the butterfly images as examples. Each image pair with the caption file in the same name,  with `.txt` or `.caption` extension.

```
data_path/butterfly
├── img711.png
├── img711.txt
├── img712.png
├── img712.txt
├── img713.png
├── img713.txt
├── img714.png
├── img714.txt
├── img715.png
└── img715.txt
```

, where the `.txt`  files describes details more than just `butterfly`.

```
a picture of a butterfly made out of paper
a picture of a yellow butterfly on a white background  
a purple butterfly with orange and pink stripes  
a colorful butterfly is shown on a white background  
a green butterfly with red and black wings  
```

Create a text file and set the extension to `.toml` and write as follow.

```toml
[general]
enable_bucket = false                        # Whether to use Aspect Ratio Bucketing

[[datasets]]
resolution = 1024                            # Training resolution
batch_size = 1                               # Batch size

  [[datasets.subsets]]
  random_crop = true												 # enable it when enable_bucket=false
  image_dir = 'data_path/butterfly'          # Specify the folder containing the training images
  caption_extension = '.txt'                 # Caption file extension; change this if using .caption
  num_repeats = 10                           # Number of repetitions for training images

  # Write the following only when using regularization images. Remove it if not using them.
  [[datasets.subsets]]
  is_reg = true
  random_crop = true												 # enable it when enable_bucket=false
  image_dir = 'path_to_regulation_dataset'   # Specify the folder containing the regularization images
  class_tokens = 'butterfly'                 # Specify the class
  num_repeats = 1                            # Number of repetitions for regularization images; 1 is usually sufficient
```



#### 3. Fine-tuning method (regularization images cannot be used)

The captions are collected in a metadata file in advance. It supports functions such as managing tags and captions separately. (Although it is called the fine-tuning method, it can also be used for non-fine-tuning.)

We have the trianing images in the `data_path/pokemon_blip/train` as 

```
data_path/pokemon_blip/train
├── img0.png
├── img1.png
├── img2.png
├── img3.png
├── ...
└── img832.png
```

Prepare a metadata file with captions and tags in JSON format with the extension `.json`.  An example of a metadata file `data_path/metadata.json` may look like

```json
{
  "img0": {
    "caption": "a drawing of a green pokemon with red eyes"
  },
  "img1": {
    "caption": "a green and yellow toy with a red nose"
  },
  "img2": {
    "caption": "a red and white ball with an angry look on its face"
  },
  ...
}
```

Create a text file and set the extension to `.toml` and write as follow.

```toml
[general]
enable_bucket = false                        

[[datasets]]
resolution = 1024                                   # Training resolution
batch_size = 1                                      # Batch size

  [[datasets.subsets]]
  random_crop = true												 
  image_dir = 'data_path/pokemon_blip/train'        # Specify the folder containing the training images
  metadata_file = '`data_path/metadata.json`'       # Metadata file name
```



### Training





### inference





### performance

