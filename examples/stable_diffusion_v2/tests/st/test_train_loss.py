import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

__dir__ = os.path.dirname(os.path.abspath(__file__))


def create_dataset(n=1):
    data_dir = __dir__ + "/demo_data"
    os.makedirs(data_dir, exist_ok=True)

    img_fn = "sunflower.png"
    img_path = __dir__ + f"/../../../videocomposer/demo_video/{img_fn}"
    caption = '"a photo of sun flowers under blue sky, vivid color"'

    shutil.copy(img_path, data_dir + f"/{img_fn}")

    tmp_annot_fp = data_dir + "/img_txt.csv"
    with open(tmp_annot_fp, "w") as fp:
        fp.write("dir,text\n")
        for i in range(n):
            if i == n - 1:
                fp.write(f"{img_path},{caption}")
            else:
                fp.write(f"{img_path},{caption}\n")

    return data_dir


@pytest.mark.parametrize("use_lora", [True, False])
def test_train_loss(use_lora):
    model_version = "sd1.5"

    # 1. create dummpy data
    data_dir = create_dataset(1)

    # 2. init vae clip with pretrained weight, init UNet randomly
    # by pop out the unet parameter from sd checkpoint

    seed = 42

    model_config = __dir__ + "/../../configs/v1-train.yaml"
    pretrained_model_path = __dir__ + "/../../models/sd_v1.5-d0ab7146.ckpt"

    output_path = __dir__
    if use_lora:
        output_path = output_path + "/lora"
        unet_initialize_random = False
    else:
        output_path = output_path + "/vanilla"
        unet_initialize_random = True

    os.makedirs(output_path, exist_ok=True)

    # export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

    cmd = (
        f"python train_text_to_image.py --data_path={data_dir} --model_config={model_config} "
        f"--pretrained_model_path={pretrained_model_path} --weight_decay=0.01 --image_size=512 "
        f"--epochs=1000 --ckpt_save_interval=600 --start_learning_rate=0.00001 --train_batch_size=1 --init_loss_scale=65536 "
        f"--use_lora={use_lora} --output_path={output_path} --warmup_steps=10 --use_ema=False --clip_grad=True --unet_initialize_random={unet_initialize_random} "
    )

    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # check ending loss

    result_log = os.path.join(output_path, "ckpt/result.log")
    df = pd.read_csv(result_log, sep="\t")  # , lineterminator='\r')
    converge_loss = np.mean(df["loss"][-100:])

    expected_loss = 0.1 if not use_lora else 0.3
    print("converge_loss: ", converge_loss)
    assert converge_loss < expected_loss


if __name__ == "__main__":
    test_train_loss(True)

    """
    result_log = __dir__ + "/../../outputs/train_lora_ovfDropUpdate_ls65536_ema_e200_revertfp32/ckpt/result.log"
    df = pd.read_csv(result_log, sep='\t') #, lineterminator='\r')
    end_loss = df['loss'][-100:]
    converge_loss = np.mean(end_loss)
    expected_loss = 0.05
    print("converge_loss: ", converge_loss)
    assert converge_loss < expected_loss
    """
