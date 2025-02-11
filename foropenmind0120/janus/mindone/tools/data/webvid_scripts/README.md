
# WebVid-10M Dataset Download and Management

## Dataset Overview

The WebVid-10M dataset is a large-scale collection of
video-text pairs collected from the Shutterstock.
The detailed dataset information can be found below:

  | Split            | # Available Videos | Storage Space |
  |------------------|--------------------|---------------|
  | WebVid-10M train | ~10.7M                  | ~20,742 GB          |
  | WebVid-10M val   | ~5K              | ~10 GB      |
  | WebVid-2M train  | ~2.5M                  | ~4,820 GB          |
  | WebVid-2m val    | ~5K              | ~9.62 GB      |


## Requirements

Ensure that you have the correct version of `video2dataset` installed before proceeding
(Version `1.3.0` will lead to an error).
```bash
pip install video2dataset==1.1.0
```

## Dividing the Dataset into Parts

Given the large size of the WebVid-10M dataset,
it is recommended to divide the CSV files into smaller parts before downloading. This can be achieved using the `divide_parts.py` script.

```bash
python divide_parts.py results_10M_train.csv
```

This command will split the `results_10M_train.csv` file into parts,
with each part containing up to 500,000 rows (by default). As a result:

- The WebVid-10M Train set will be divided into 23 parts.
- The WebVid-2M Train set will be divided into 6 parts.

## Downloading the Dataset

To download a specific part of the dataset, run:

```bash
bash download_webvid.sh 10M_train part0 data1
```

Replace `10M_train`, `part0`, and `data1` with the appropriate dataset type, part number, and output directory.

**Note:** The download process may occasionally fail.
Manually check the creation time of the temporary file to see if it is still downloading.
If the download is not progressing, restart the terminal and rerun the command.
`video2dataset` by default supports incremental downloading.

## Fixing CSV Files

During the download process, some CSV files might not be
read correctly due to entries being split across multiple
lines.

To fix this, you can run

```bash
python fix_csv.py --root ./webvid-10m --set 2M_train --part 0
```

This will process `part0.csv` in the `2M_train` set and
save a corrected version as `part0_fixed.csv`.

## Verifying the Download

You can verify the completeness of the download
using the `datacheck_webvid.py` script.
This script will check the number of videos downloaded
and report the success rate. Example command:

```bash
python datacheck_webvid.py --set 10M_train --part 0 --root ./webvid-10m
```

This will check the downloaded files in `part0` of
the `10M_train` set.
