# Download

## Download Datasets
### Requirements
1. Use python>=3.8 [[install]](https://www.python.org/downloads/)
2. Install `huggingface_hub>=0.23.5`
3. Install `pandas`

### Datasets Information
The [Open-Sora-Dataset-v1.1.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main) includes three image-text datasets and three video-text datasets. As reported in [Report v1.1.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.1.0.md), the three image-text datasets are:
| Name | Image Source | Text Captioner | Num pair |
|---|---|---|---|
| SAM-11M | [SAM](https://ai.meta.com/datasets/segment-anything/) |  [LLaVA](https://github.com/haotian-liu/LLaVA) |  11,185,255 |
| Anytext-3M-en | [Anytext](https://github.com/tyxsspa/AnyText) |  [InternVL-1.5](https://github.com/OpenGVLab/InternVL) |  1,886,137 |
| Human-160k | [Laion](https://laion.ai/blog/laion-5b/) |  [InternVL-1.5](https://github.com/OpenGVLab/InternVL) |  162,094 |


The three video-text datasets are:
| Name | Hours | Num frames | Num pair |
|---|---|---|---|
| [Mixkit](https://mixkit.co/) | 42.0h |  65 |  54,735 |
|   |  |  513 |  1,997 |
| [Pixabay](https://pixabay.com/) | 353.3h |  65 | 601,513 |
|   |  |  513 |  51,483 |
| [Pexel](https://www.pexels.com/) | 2561.9h |  65 |  3,832,666 |
|   |  |  513 |  271,782 |

### Downloading scripts
Among the six datasets, `Mixkit`, `Pixabay`, `Pexel`, `Human-160k` can be downloaded from [OpenSora-PKU-v1.1.0 huggingface webpage](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main). For automatic downloading, please refer to [`tools/download/download_HF_datasets.py`](download_HF_datasets.py).

`SAM-11M` is also named as Segment Anything 1 Billion (SA-1B) dataset. To download this dataset, please follow the instruction of [Meta webpage](https://ai.meta.com/datasets/segment-anything-downloads/) and download the text file that records the file names and links. Afterwards, you can run [`tools/download/download_sam.py`](./download_sam.py) to download this dataset.

`Anytext-3M-en` dataset, according to the [official github page](https://github.com/tyxsspa/AnyText), can be downloaded from [ModelScope](https://modelscope.cn/datasets/iic/AnyText-benchmark/files) or [GoogleDrive](https://drive.google.com/drive/folders/1Eesj6HTqT1kCi6QLyL5j0mL_ELYRp3GV). We recommend to download it via the [GoogleDrive](https://drive.google.com/drive/folders/1Eesj6HTqT1kCi6QLyL5j0mL_ELYRp3GV) link.
