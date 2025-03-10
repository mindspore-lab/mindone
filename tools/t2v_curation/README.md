# Video-Text Data Processing Pipeline
>Automatic T2V HQ Data Curation Pipeline v1.0 MindSpore version.
>
<img width="850" alt="data_pipeline_demo" src="https://github.com/user-attachments/assets/a7828e69-59b9-49f2-b762-a50ac5bd86a1" />

## Overview
This pipeline is designed to gather video-text pairs to train video generation models 
based on text inputs. 

First, raw videos — whether sourced from the internet or public 
datasets — are divided into shorter clips using scene detection 
techniques. We offer an optional filtering mechanism to select 
specific video categories of interest. Following this, we incorporate 
`imagededup` package to remove duplicate or near duplicate videos from the dataset.

Next, these videos undergo an evaluation process where multiple 
scores are predicted using existing models. These scores include 
aesthetic scoring, OCR (Optical Character Recognition) for text 
detection, and LPIPS scoring to assess motion. 
Only videos that meet satisfactory evaluation criteria advance 
to the captioning step.

After captioning, a matching score is calculated to assess the 
alignment between video and text. Samples with low matching scores
are filtered out.

In summary, our pipeline generates video-text pairs that exhibit 
high aesthetic quality, significant video motion, and strong 
semantic consistency. You may refer to the 
[Further Reading](#further-reading) section for more details.

## Requirement
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Example Workflow

### Configuration

The pipeline is configured using a `config.yaml` file located
in the `config/` directory. This file allows you to specify paths, 
enable or disable pipeline steps, and set parameters for each
processing stage.

#### Set Root Paths
In `config.yaml`, modify the following paths to match your 
directory structure:

```bash
paths:
  ROOT_VIDEO: "/path/to/video/folder" # Directory containing the original video files.
  ROOT_CLIPS: "/path/to/video/clips/folder" # Directory where video clips will be stored.
  ROOT_META: "/path/to/meta/folder" # Directory for metadata CSV files.
```

#### Deduplication Setup
If you need to perform deduplication, run the following command:
```bash
python pipeline/datasets/imagededup/setup.py build_ext --inplace
```

#### Scoring Model Setup
If aesthetic scoring or CLIP matching is needed, download the models
and set up according to the guideline [here](./pipeline/scoring/README.md).

#### Captioning Model Setup
Follow the guideline [here](./pipeline/captioning/README.md).

#### Customize Pipeline Steps
Enable or disable specific pipeline steps via setting `run`
and adjust their parameters as needed. We recommend keeping
the default for most parts. If you are interested in option
filtering, you may set `run` under `option_filtering` to `true`
and provide the video type you would like to keep under `option`.

### Usage

#### Via Config and Runner

We support using json or yaml config files. After setting the `config.yaml` or `config.json` file, run the entire pipeline via
```bash
python -m script.pipeline_runner ./config/config.yaml
```

OR

```bash
python -m script.pipeline_runner ./config/config.json
```

You will get the processed csv file containing metadata information
after running the pipeline. We also store the intermediate csv files
in between during processing.

#### Step by Step

You may also run the pipeline step by step or run certain steps
based on your needs. Refer to [Command Line Workflow](./cmd_guide.md)
for more details.

## Further Reading
For more information, please refer to:
- [Dataset Management](./pipeline/datasets/README.md)
- [Scene Detection and Video Splitting](./pipeline/splitting/README.md)
- [Scoring and Filtering](./pipeline/scoring/README.md)
- [Captioning](./pipeline/captioning/README.md)

## Acknowledgement
This pipeline for video/image data processing pipeline in MindSpore is mostly 
based on the [work](https://github.com/hpcaitech/Open-Sora/blob/main/docs/data_processing.md) by HPC-AI OpenSora. We thank them for their generous
support of the open source community.
