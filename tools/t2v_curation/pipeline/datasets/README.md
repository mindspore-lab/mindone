# Dataset Management

- [Dataset Management](#dataset-management)
  - [Dataset Format](#dataset-format)
  - [Dataset to CSV](#dataset-to-csv)
  - [Manage datasets](#manage-datasets)
    - [Basic Usage](#basic-usage)
    - [Score filtering](#score-filtering)
    - [Documentation](#documentation)
  - [Analyze datasets](#analyze-datasets)

You can use the following commands to manage the dataset.

## Dataset to CSV

As a starting point, `convert.py` is used to convert
the dataset (i.e., videos or images in a folder) to a CSV file. You can use the following
commands to convert the dataset to a CSV file:

```bash
# general video folder
python -m pipeline.datasets.convert video VIDEO_FOLDER --output video.csv
# general image folder
python -m pipeline.datasets.convert image IMAGE_FOLDER --output image.csv
```

## Dataset Format

All data should be tracked in a `.csv` file (or `parquet.gzip` to save space), which is used for both training and data preprocessing. The columns should follow the words below:

- `path`: the relative/absolute path or url to the image or video file.
- `text`: the caption or description of the image or video.
- `num_frames`: the number of frames in the video.
- `width`: the width of the video frame.
- `height`: the height of the video frame.
- `aspect_ratio`: the aspect ratio of the video frame (height / width).
- `resolution`: height x width. For analysis.
- `text_len`: the number of tokens in the text. For analysis.
- `aes`: aesthetic score calculated by the asethetic scorer based on CLIP. For filtering.
- `match`: matching score of an image-text/video-text pair calculated by CLIP. For filtering.
- `lpips`: LPIPS score calculated by extracting frames from the video at regular intervals and calculating the perceptual similarity between consecutive frames. -1 if only one or no frame is extracted.
- `nsfw`: NSFW flag, 1 for not NSFW, 0 for NSFW.
- `fps`: the frame rate of the video.


An example ready for training:

```csv
path, text, num_frames, width, height, aspect_ratio
/absolute/path/to/image1.jpg, caption, 1, 720, 1280, 0.5625
/absolute/path/to/video1.mp4, caption, 120, 720, 1280, 0.5625
/absolute/path/to/video2.mp4, caption, 20, 256, 256, 1
```

## Manage datasets

We use `datautil.py` to manage the dataset.

### Basic Usage

You can use the following commands to process the `.csv`
or `.parquet` files. The output file will be saved in the same directory as the input, with different suffixes indicating the processed method.

```bash
# datautil takes multiple CSV files as input and merges them into one CSV file
# output: DATA1+DATA2.csv
python -m pipeline.datasets.datautil DATA1.csv DATA2.csv

# shard CSV files into multiple CSV files
# output: DATA1_0.csv, DATA1_1.csv, ...
python -m pipeline.datasets.datautil DATA1.csv --shard 10

# filter frames between 128 and 256
# output: DATA1_fmin_128_fmax_256.csv
python -m pipeline.datasets.datautil DATA.csv --fmin 128 --fmax 256

# Disable parallel processing
python -m pipeline.datasets.datautil DATA.csv --fmin 128 --fmax 256 --disable-parallel

# Compute num_frames, height, width, fps, aspect_ratio for videos or images
# output: IMG_DATA+VID_DATA_vinfo.csv
python -m pipeline.datasets.datautil IMG_DATA.csv VID_DATA.csv --video-info

# You can run multiple operations at the same time.
python -m pipeline.datasets.datautil DATA.csv --video-info --remove-empty-caption --remove-url --lang en
```

### Score Filtering

To examine and filter the dataset by
aesthetic score or CLIP matching score,
you can use the following commands:

```bash
# sort the dataset by aesthetic score
# output: DATA_sort.csv
python -m pipeline.datasets.datautil DATA.csv --sort aesthetic_score
# View examples of high aesthetic score
head -n 10 DATA_sort.csv
# View examples of low aesthetic score
tail -n 10 DATA_sort.csv

# sort the dataset by clip score
# output: DATA_sort.csv
python -m pipeline.datasets.datautil DATA.csv --sort clip_score

# filter the dataset by aesthetic score
# output: DATA_aesmin_0.5.csv
python -m pipeline.datasets.datautil DATA.csv --aesmin 0.5
# filter the dataset by clip score
# output: DATA_matchmin_0.5.csv
python -m pipeline.datasets.datautil DATA.csv --matchmin 0.5
```

### Documentation

You can also use `python -m pipeline.datasets.datautil --help` to see usage.

| Args                         | File suffix     | Description                                                    |
|------------------------------|-----------------|----------------------------------------------------------------|
| `--output OUTPUT`            |                 | Output path                                                    |
| `--format FORMAT`            |                 | Output format (csv, parquet)                     |
| `--disable-parallel`         |                 | Disable `pandarallel`                                          |
| `--seed SEED`                |                 | Random seed                                                    |
| `--shard SHARD`              | `_0`,`_1`, ...  | Shard the dataset                                              |
| `--sort KEY`                 | `_sort`         | Sort the dataset by KEY in descending order                    |
| `--sort_ascending KEY`       | `_sort`         | Sort the dataset by KEY in ascending order                     |
| `--difference DATA.csv`      |                 | Remove the paths in DATA.csv from the dataset                  |
| `--intersection DATA.csv`    |                 | Keep the paths in DATA.csv from the dataset and merge columns  |
| `--info`                     | `_info`         | Get the basic information of each video and image (cv2)        |
| `--ext`                      | `_ext`          | Remove rows if the file does not exist                         |
| `--relpath`                  | `_relpath`      | Modify the path to relative path by root given                 |
| `--abspath`                  | `_abspath`      | Modify the path to absolute path by root given                 |
| `--remove-empty-caption`     | `_noempty`      | Remove rows with empty caption                                 |
| `--remove-url`               | `_nourl`        | Remove rows with url in caption                                |
| `--lang LANG`                | `_lang`         | Remove rows with other language                                |
| `--remove-path-duplication`  | `_noduppath`    | Remove rows with duplicated path                               |
| `--remove-text-duplication`  | `_noduptext`    | Remove rows with duplicated caption                            |
| `--refine-llm-caption`       | `_llm`          | Modify the caption generated by LLM                            |
| `--clean-caption`            | `_clean`        | Modify the caption according to T5 pipeline to suit training   |
| `--merge-cmotion`            | `_cmotion`      | Merge the camera motion to the caption                         |
| `--load-caption EXT`         | `_load`         | Load the caption from the file                                 |
| `--fmin FMIN`                | `_fmin`         | Filter the dataset by minimum number of frames                 |
| `--fmax FMAX`                | `_fmax`         | Filter the dataset by maximum number of frames                 |
| `--hwmax HWMAX`              | `_hwmax`        | Filter the dataset by maximum height x width                   |
| `--aesmin AESMIN`            | `_aesmin`       | Filter the dataset by minimum aesthetic score                  |
| `--ocr_box_max BOXMAX`       | `_ocrboxmax`    | Filter the dataset by maximum total number of text boxes       |
| `--ocr_single_max SINGLEMAX` | `_ocrsinglemax` | Filter the dataset by maximum single text box area percentage  |
| `--ocr_total_max TOTALMAX`   | `_ocrtotalmax`  | Filter the dataset by maximum total text boxes area percentage |
| `--matchmin MATCHMIN`        | `_matchmin`     | Filter the dataset by minimum clip score                       |
| `--lpipsmin LPIPSMIN`        | `_lpipsmin`     | Filter the dataset by minimum LPIPS score                      |
| `--safety_check`             | `_safe`         | Filter out the videos flagged NSFW                             |
