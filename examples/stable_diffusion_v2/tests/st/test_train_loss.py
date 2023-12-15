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


@pytest.mark.parametrize("use_lora", [True, False])  # 'db'
def test_vanilla_lora(use_lora):
    # model_version = "sd1.5"
    # seed = 42

    # 1. create dummpy data
    data_dir = create_dataset(1)

    # 2. init vae clip with pretrained weight, init UNet randomly
    # by pop out the unet parameter from sd checkpoint

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
        f"--pretrained_model_path={pretrained_model_path} --weight_decay=0.01 --image_size=512 --dataset_sink_mode=True "
        f"--epochs=1000 --ckpt_save_interval=1000 --start_learning_rate=0.00001 --train_batch_size=1 --init_loss_scale=65536 "
        f"--use_lora={use_lora} --output_path={output_path} --warmup_steps=10 --use_ema=False --clip_grad=True "
        f"--unet_initialize_random={unet_initialize_random} "
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


def test_db():
    # model_version = "sd1.5"
    # seed = 42

    # 1. create dummpy data
    data_dir = create_dataset(1)
    # data_dir = __dir__ + "/../../datasets/dog"

    train_config = __dir__ + "/config/train_config_dreambooth_v1.yaml"
    # train_config = __dir__ + "/../../configs/train/train_config_dreambooth_v1.yaml"
    pretrained_model_path = __dir__ + "/../../models/sd_v1.5-d0ab7146.ckpt"
    # instance_prompt = "a photo of sks sunflower"
    # class_prompt = "a photo of a sunflower"
    # class_data_dir = "temp_class_images/dog"
    class_data_dir = "temp_class_images/sunflower"

    output_path = __dir__ + "/db"
    os.makedirs(output_path, exist_ok=True)

    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

    epochs = 100
    cmd = (
        f"python train_dreambooth.py "
        f"--train_config {train_config} "
        f"--instance_data_dir {data_dir} "
        f"--class_data_dir {class_data_dir} "
        f"--output_path  {output_path} "
        f"--pretrained_model_path {pretrained_model_path} "
        f"--unet_initialize_random True "
        f"--epochs={epochs} --ckpt_save_interval={epochs} --num_class_images=200 --dataset_sink_mode=True "  # 800 steps
    )

    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # check ending loss
    result_log = os.path.join(output_path, "ckpt/rank_0/result.log")
    df = pd.read_csv(result_log, sep="\t")  # , lineterminator='\r')
    ending = max(1, epochs // 10)
    converge_loss = np.mean(df["loss"][-ending:])  # 200 steps

    expected_loss = 1.0
    print("converge_loss: ", converge_loss)  #
    assert converge_loss < expected_loss


def run_task(task="vanilla"):
    if task == "vanilla":
        test_vanilla_lora(use_lora=False)
    elif task == "lora":
        test_vanilla_lora(use_lora=True)
    elif task == "db":
        test_db()
    else:
        raise ValueError


if __name__ == "__main__":
    from fire import Fire

    Fire(run_task)

    # test_vanilla_lora(False)
    # test_db()
