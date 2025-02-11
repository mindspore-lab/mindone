Performance Boosting History

In this doc, we record the performance history of Open-Sora-Plan v1.1.0.
History:



- 2024.07.16
  Major Updates ([PR#596](https://github.com/mindspore-lab/mindone/pull/596)):
  1. Update CANN version ([CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)->[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)) and MindSpore version ([MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) -> [MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/));
  2. Enable optimizer parallelism fusion by default (See `enable_parallel_fusion` in train args);
  3. Change FlashAttention default layout from `BNSD` to `BSH` (See `mindone.models.modules.flash_attention`);
  4. Set `num_no_recompute` to $4$ in `scripts/text_condition/train_videoae_65x512x512_img16.sh`.

| Model           | Context        | Precision | BS  | NPUs | num_frames + num_images | Resolution  | Train T. (s/step) |
|:----------------|:---------------|:----------|:---:|:----:|:-----------------------:|:-----------:|:-----------------:|
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  2  |  8   |         17 + 4          | 512x512     |       2.45        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  2  |  8   |         65 + 16         | 512x512     |       9.36       |
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  2  |  8   |         65 + 4          | 512x512     |       7.02        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  1  |  8   |         221 + 4         | 512x512     |       7.18        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  1  |  8   |         513 + 8         | 512x512     |        12.3       |

> Context: {NPU type}-{CANN version}-{MindSpore version}

- 2024.07.03

Initial performance record:

| Model           | Context        | Precision | BS  | NPUs | num_frames + num_images | Resolution  | Train T. (s/step) |
|:----------------|:---------------|:----------|:---:|:----:|:-----------------------:|:-----------:|:-----------------:|
| LatteT2V-XL/122 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) | BF16      |  2  |  8   |         17 + 4          | 512x512     |       2.54        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) | BF16      |  2  |  8   |         65 + 16         | 512x512     |       10.57       |
| LatteT2V-XL/122 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) | BF16      |  2  |  8   |         65 + 4          | 512x512     |       7.50        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) | BF16      |  1  |  8   |         221 + 4         | 512x512     |       7.18        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) | BF16      |  1  |  8   |         513 + 8         | 512x512     |       12.5        |

> Context: {NPU type}-{CANN version}-{MindSpore version}
