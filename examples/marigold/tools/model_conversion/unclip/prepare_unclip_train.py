#!/usr/bin/env python
# Prepare the unclip pretraining checkpoint. It is a combination of `sd21-unclip-l` and `sd_v2_768_v`
# Currectly the unclip training/finetune can not be started directly from `sd21-unclip-l` since
# the loss will be easily overflowed/diverged for unkown reason under MS2.0/2.1
import os
import sys
from os.path import abspath, dirname

import mindspore as ms

root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.insert(0, root_dir)

from utils.download import download_checkpoint

_URL_PREFIX = "https://download.mindspore.cn/toolkits/mindone/stable_diffusion"

UNCLIP_CKPT = "sd21-unclip-l-baa7c8b5.ckpt"
SD_V2_768 = "sd_v2_768_v-e12e3a9b.ckpt"


def main():
    download_checkpoint(os.path.join(_URL_PREFIX, UNCLIP_CKPT), "models")
    download_checkpoint(os.path.join(_URL_PREFIX, SD_V2_768), "models")

    # load unclip checkpoint
    content = ms.load_checkpoint(os.path.join("models", UNCLIP_CKPT))
    add_content = dict()
    for k, v in content.items():
        if "embedder." in k or "cond_stage_model." in k or "model.diffusion_model.label_emb." in k:
            add_content[k] = v

    # load sd_v2_768 checkpoint
    content = ms.load_checkpoint(os.path.join("models", SD_V2_768))
    new_content = dict()
    for k, v in content.items():
        if "cond_stage_model." not in k:
            new_content[k] = v

    # merge them together and write to model/
    new_content.update(add_content)
    record = list()
    for k, v in new_content.items():
        record.append({"name": k, "data": v})

    ms.save_checkpoint(record, os.path.join("models", "sd_v2_v_embedder.ckpt"))


if __name__ == "__main__":
    main()
