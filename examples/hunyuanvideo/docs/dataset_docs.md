# Dataset Prepation

A dataset of videos and corresponding captions is required for training. Therefore two inputs are necessary:
1. An input video folder that contains all the videos to be used for training.
2. A dataset file (.csv file) that specifis the video paths and the captions.

An example of csv file is shown below:

|video|caption|
| ---|---|
|Dogs/dog1.mp4|A dog is running on the grass.|
|Cats/cat1.mp4|A cat is sitting on the grass.|

Note that the video paths in the csv file are relative to the input video folder.

You should prepare two csv files: the training csv file and the validation csv file, if you need to run validation during training.

# Text Embeding Cache

Text embedding cache is needed for both the training and validation dataset, please run:
```bash
python scripts/run_text_encoder.py \
  --data-file-path /path/to/train_caption.csv \
  --output-path /path/to/text_embed_folder \

python scripts/run_text_encoder.py \
  --data-file-path /path/to/val_caption.csv \
  --output-path /path/to/text_embed_folder \
```

Please also extract the text embedding for an empty string, because it will be used during training when the prompt is dropped randomly.
```bash
python scripts/run_text_encoder.py \
  --prompt "" \
  --output-path /path/to/text_embed_folder \
```

# Training Parameters

In configuration file, such as `configs/train/stage1_t2v_256px.yaml`, you can set the following parameters respective to the training dataset:
```yaml
dataset:
  csv_path: CSV_PATH  # train csv file path
  video_folder: VIDEO_FOLDER  # train video folder
  text_emb_folder: TEXT_EMB_FOLDER  # train text embedding folder
  empty_text_emb: EMPTY_TEXT_EMB_PATH  # empty text embedding path (.npz)
```
You can also set the following parameters respective to the validation dataset:
```yaml
valid:
  dataset:
    csv_path: CSV_PATH # validation csv file path
    video_folder: VIDEO_FOLDER # validation video folder
    text_emb_folder: TEXT_EMB_FOLDER # validation text embedding folder
```

[Here](../scripts/hyvideo/train_t2v_zero3.sh) is an example of training script which sets the dataset parameters using command line arguments.
