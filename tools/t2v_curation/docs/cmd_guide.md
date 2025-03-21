## Command Line Guideline

Below is a sample example command line workflow.
You may run the pipeline step by step or run certain steps
based on your needs.


### 0. set up
```bash
ROOT_VIDEO="/path/to/video/folder"
ROOT_CLIPS="/path/to/video/clips/folder"
ROOT_META="/path/to/meta/folder"
export PYTHONPATH=$(pwd)
# run the command below to set up deduplication if needed
python pipeline/datasets/imagededup/setup.py build_ext --inplace
```

### 1. Convert dataset to CSV
**1.1 Create a meta file from a video folder.**
```bash
python -m pipeline.datasets.convert video ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv
```

**1.2 Get video information and remove broken videos.**
```bash
python -m pipeline.datasets.datautil ${ROOT_META}/meta.csv --info --fmin 1
```

### 2. Split video to clips
**2.1 Detect scenes.**
```bash
python -m pipeline.splitting.scene_detect ${ROOT_META}/meta_info_fmin1.csv
```

**2.2 Cut video into clips based on scenes. This should produce video clips under `${ROOT_CLIPS}`.**
```bash
python -m pipeline.splitting.cut ${ROOT_META}/meta_info_fmin1_timestamp.csv --save_dir ${ROOT_CLIPS}
```

**2.3 Create a meta file for video clips.**
```bash
python -m pipeline.datasets.convert video ${ROOT_CLIPS} --output ${ROOT_META}/meta_clips.csv
```

**2.4 Get clips information and remove the broken ones.**
```bash
python -m pipeline.datasets.datautil ${ROOT_META}/meta_clips.csv --info --fmin 1
```

### 3. Deduplication
```bash
python -m pipeline.datasets.deduplication ${ROOT_META}/meta_clips_info_fmin1.csv
```

### 4. Scoring and filtering
For convenience, we assume `working_meta.csv` is the input file
under the `${ROOT_META}` directory for all commands below.

**4.1.1 Calculate matching scores with an option.**
```
python -m pipeline.scoring.matching.inference ${ROOT_META}/working_meta.csv --option animal --use_cpu # cpu
```
```bash
# modify worker_num and local_worker_num based on your resource, same below
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/matching/inference.py \
 ${ROOT_META}/working_meta.csv --option animal # Ascend
```

**4.1.2 Filter videos based on an option.**
```bash
python -m pipeline.datasets.datautil ${ROOT_META}/working_meta.csv --matchmin 20
```

**4.2.1 Perform OCR recognition.**
```bash
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/ocr/inference.py \
 /path/to/meta.csv \
 --total_text_percentage
```

**4.2.2 Filter videos based on total text percentage.**
```bash
python -m pipeline.datasets.datautil ${ROOT_META}/working_meta.csv --ocr_total_max 0.1
```

**4.3.1 Predict LPIPS scores.**
```bash
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/lpips/inference.py \
 ${ROOT_META}/working_meta.csv # Ascend
```

**4.3.2 Filter videos based on LPIPS scores.**
```bash
pyhton -m pipeline.datasets.datautil ${ROOT_META}/working_meta.csv --lpipsmin 0.2
```

**4.4.1 Predict aesthetic scores.**
```bash
python -m scoring.aesthetic.inference ${ROOT_META}/working_meta.csv --use_cpu # cpu
```

```bash
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/aesthetic/inference.py \
 ${ROOT_META}/working_meta.csv # Ascend
```

**4.4.2 Filter by aesthetic scores.**
```bash
python -m pipeline.datasets.datautil ${ROOT_META}/working_meta.csv --aesmin 4.5
```

**4.5.1 Determine whether the video is NSFW.**
```bash
python -m scoring.nsfw.inference ${ROOT_META}/working_meta.csv --use_cpu # cpu
```

```bash
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/nsfw/inference.py \
 ${ROOT_META}/working_meta.csv # Ascend
```

**4.5.2 Filter out videos flagged NSFW.**
```bash
python -m pipeline.datasets.datautil ${ROOT_META}/working_meta.csv --safety_check
```

### 5. Captioning and calculating matching scores
**5.1 Generate caption.**
```bash
# Qwen2-VL caption
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/captioning/caption_qwen2vl.py \
 ${ROOT_META}/working_meta.csv # support Ascend only
```

```bash
# LLaVA caption
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/captioning/caption_llava.py \
 ${ROOT_META}/working_meta.csv # support Ascend only
```

```bash
# PLLaVA caption
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/captioning/caption_pllava.py \
 ${ROOT_META}/working_meta.csv # support Ascend only
```

**5.2 Clean caption.**
```bash
python -m pipeline.datasets.datautil ${ROOT_META}/working_meta.csv \
 --clean-caption --refine-llm-caption --remove-empty-caption
```

**5.3 Calculate matching scores with captions.**
```bash
python -m pipeline.scoring.matching.inference ${ROOT_META}/working_meta.csv --use_cpu # cpu
```
```bash
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/matching/inference.py \
 ${ROOT_META}/working_meta.csv # Ascend
```

**5.4 Filter by matching scores.**
```bash
python -m pipeline.datasets.datautil ${ROOT_META}/working_meta.csv --matchmin 20
```
