"""
vanilla_8p, unet randomly initialized, loss change 1.0 ->  0.3
Vanilla, unet randomly initialized, loss change 1.0 ->  0.1
LoRA, loss not stable, but 0.5 is a safe threshold
Dreambooth, unet randomly initialized, loss  change 2.0 -> 0.5
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from _common import down_checkpoint


def create_dataset(n=1):
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "demo_data")
    os.makedirs(data_dir, exist_ok=True)

    img_fn = "sunflower.png"
    img_path = os.path.join(root, f"../../../videocomposer/demo_video/{img_fn}")
    caption = '"a photo of sunflowers under blue sky, vivid color"'

    shutil.copy(img_path, os.path.join(data_dir, f"{img_fn}"))

    tmp_annot_fp = os.path.join(data_dir, "img_txt.csv")
    with open(tmp_annot_fp, "w") as fp:
        fp.write("dir,text\n")
        for i in range(n):
            if i == n - 1:
                fp.write(f"{img_path},{caption}")
            else:
                fp.write(f"{img_path},{caption}\n")

    return data_dir


@pytest.mark.parametrize("version", ["1.5", "2.0"])
def test_vanilla_8p(version):
    root = os.path.dirname(os.path.abspath(__file__))
    expected_loss = 0.3

    # 1. create dummpy data
    data_dir = os.path.join(root, "../../datasets/chinese_art_blip/train")

    # 2. init vae clip with pretrained weight, init UNet randomly
    # by pop out the unet parameter from sd checkpoint
    if version == "1.5":
        train_config = os.path.join(root, "../../configs/train/train_config_vanilla_v1.yaml")
        pretrained_model_path = os.path.join(root, "../../models/sd_v1.5-d0ab7146.ckpt")
        if not os.path.exists(pretrained_model_path):
            pretrained_model_path = down_checkpoint(version=version)
    elif version == "2.0":
        train_config = os.path.join(root, "../../configs/train/train_config_vanilla_v2.yaml")
        pretrained_model_path = os.path.join(root, "../../models/sd_v2_base-57526ee4.ckpt")
        if not os.path.exists(pretrained_model_path):
            pretrained_model_path = down_checkpoint(version=version)
    else:
        raise ValueError(f"SD {version} not included in test")

    output_path = root
    output_path = os.path.join(output_path, "vanilla_8p")

    os.makedirs(output_path, exist_ok=True)

    # export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"  # It depends on MS version
    epochs = 300

    cmd = (
        f"mpirun --allow-run-as-root -n 8 python train_text_to_image.py --data_path={data_dir} --train_config={train_config} "
        f"--pretrained_model_path={pretrained_model_path} --dataset_sink_mode=True --callback_size=3 --scheduler=constant "
        f"--epochs={epochs} --ckpt_save_interval={epochs} --init_loss_scale=65536 --image_size=256 --start_learning_rate=1e-4 "
        f"--output_path={output_path} --clip_grad=False --unet_initialize_random=True --use_parallel=True --warmup_steps=50 "
    )

    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # check ending loss
    result_log = os.path.join(output_path, "result.log")
    df = pd.read_csv(result_log, sep="\t")  # , lineterminator='\r')
    converge_loss = np.mean(df["loss"][-100:])

    print("converge_loss: ", converge_loss)
    assert converge_loss < expected_loss


@pytest.mark.parametrize("use_lora", [True, False])  # lora or vanilla
@pytest.mark.parametrize("version", ["1.5", "2.0"])
def test_vanilla_lora(use_lora, version):
    root = os.path.dirname(os.path.abspath(__file__))
    expected_loss = 0.1 if not use_lora else 0.5

    # 1. create dummpy data
    data_dir = create_dataset(5)

    # 2. init vae clip with pretrained weight, init UNet randomly
    # by pop out the unet parameter from sd checkpoint
    if version == "1.5":
        model_config = os.path.join(root, "../../configs/v1-train.yaml")
        pretrained_model_path = os.path.join(root, "../../models/sd_v1.5-d0ab7146.ckpt")
        if not os.path.exists(pretrained_model_path):
            pretrained_model_path = down_checkpoint(version=version)
        infer_config = os.path.join(root, "../../configs/v1-inference.yaml")
    elif version == "2.0":
        model_config = os.path.join(root, "../../configs/v2-train.yaml")
        pretrained_model_path = os.path.join(root, "../../models/sd_v2_base-57526ee4.ckpt")
        if not os.path.exists(pretrained_model_path):
            pretrained_model_path = down_checkpoint(version=version)
        infer_config = os.path.join(root, "../../configs/v2-inference.yaml")
    else:
        raise ValueError(f"SD {version} not included in test")

    output_path = root
    if use_lora:
        output_path = os.path.join(output_path, "lora")
        unet_initialize_random = False
        start_learning_rate = 0.0001
    else:
        output_path = os.path.join(output_path, "vanilla")
        unet_initialize_random = True
        start_learning_rate = 0.00001

    os.makedirs(output_path, exist_ok=True)

    # export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"  # It depends on MS version
    epochs = 300

    cmd = (
        f"python train_text_to_image.py --data_path={data_dir} --model_config={model_config} "
        f"--pretrained_model_path={pretrained_model_path} --weight_decay=0.01 --image_size=256 --dataset_sink_mode=True --callback_size=5 "
        f"--epochs={epochs} --ckpt_save_interval={epochs} --start_learning_rate={start_learning_rate} --train_batch_size=1 --init_loss_scale=65536 "
        f"--use_lora={use_lora} --output_path={output_path} --warmup_steps=10 --use_ema=False --clip_grad=False "
        f"--unet_initialize_random={unet_initialize_random} "
    )

    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # check ending loss
    result_log = os.path.join(output_path, "result.log")
    df = pd.read_csv(result_log, sep="\t")  # , lineterminator='\r')
    converge_loss = np.mean(df["loss"][-100:])

    print("converge_loss: ", converge_loss)
    assert converge_loss < expected_loss
    # test inference
    end_ckpt = os.path.join(output_path, "ckpt", f"sd-{epochs}.ckpt")
    if use_lora:
        lora_ckpt_path = end_ckpt
        ckpt_path = pretrained_model_path
        cmd = (
            f"python text_to_image.py --config={infer_config} --n_iter=1 --n_samples=2 "
            f"--output_path={output_path} --lora_ckpt_path={lora_ckpt_path} --use_lora={use_lora} "
            f"--ckpt_path={ckpt_path} --negative_prompt='sunflowers' "
        )
        print(f"Running command: \n{cmd}")
        ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
        assert ret == 0, "run text_to_image.py fails"


@pytest.mark.parametrize("version", ["1.5", "2.0"])
def test_db(version):
    # seed = 42
    root = os.path.dirname(os.path.abspath(__file__))
    # 1. create dummpy data
    data_dir = create_dataset(1)
    if version == "1.5":
        train_config = os.path.join(root, "../../configs/train/train_config_dreambooth_v1.yaml")
        pretrained_model_path = os.path.join(root, "../../models/sd_v1.5-d0ab7146.ckpt")
    elif version == "2.0":
        train_config = os.path.join(root, "../../configs/train/train_config_dreambooth_v2.yaml")
        pretrained_model_path = os.path.join(root, "../../models/sd_v2_base-57526ee4.ckpt")
    else:
        raise ValueError(f"SD {version} not included in test")

    class_data_dir = "temp_class_images/sunflower"

    output_path = os.path.join(root, "db")
    os.makedirs(output_path, exist_ok=True)

    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

    epochs = 50
    cmd = (
        f"python train_dreambooth.py "
        f"--train_config {train_config} "
        f"--instance_data_dir {data_dir} "
        f"--class_data_dir {class_data_dir} "
        f"--output_path  {output_path} "
        f"--pretrained_model_path {pretrained_model_path} "
        f"--unet_initialize_random True "
        f"--image_size=256 --callback_size=40 --dataset_sink_mode=True "
        f"--epochs={epochs} --ckpt_save_interval={epochs} --num_class_images=200 "
        f"--with_prior_preservation=False "  # 800 steps
    )

    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # check ending loss
    result_log = os.path.join(output_path, "rank_0/result.log")
    df = pd.read_csv(result_log, sep="\t")  # , lineterminator='\r')
    ending = max(1, epochs // 10)
    converge_loss = np.mean(df["loss"][-ending:])  # 200 steps

    expected_loss = 0.5
    print("converge_loss: ", converge_loss)  #
    assert converge_loss < expected_loss


def run_task(task="vanilla", version="1.5", device_num=1):
    if task == "vanilla" and device_num == 1:
        test_vanilla_lora(use_lora=False, version=version)
    elif task == "lora" and device_num == 1:
        test_vanilla_lora(use_lora=True, version=version)
    elif task == "db" and device_num == 1:
        test_db(version=version)
    elif task == "vanilla" and device_num == 8:
        test_vanilla_8p(version=version)
    else:
        raise ValueError("please check task, version and device_num")


if __name__ == "__main__":
    # test_vanilla_8p("1.5")
    # test_vanilla_lora(False, "1.5")
    # test_vanilla_lora(True, "1.5")
    # test_db("1.5")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="vanilla", choices=["vanilla", "lora", "db"], help="stable diffusion task"
    )
    parser.add_argument("--version", type=str, default="1.5", choices=["1.5", "2.0"], help="stable diffusion version")
    parser.add_argument(
        "--device_num",
        type=int,
        default=1,
        choices=[1, 8],
        help="device num, only supports 1 or 8. Note: device_num=8 only supports vanilla task.",
    )
    args = parser.parse_args()
    run_task(task=args.task, version=args.version, device_num=args.device_num)
