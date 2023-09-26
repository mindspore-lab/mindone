import os
import subprocess
import sys

import pytest
from _common import gen_dummpy_data

gen_dummpy_data()


@pytest.mark.parametrize("use_ema", [True])
@pytest.mark.parametrize("finetuning", ["Vanilla", "LoRA"])
def test_train_infer(use_ema, finetuning):
    data_path = "data/Canidae/val/dogs"
    model_config_file = "configs/v2-train.yaml"
    pretrained_model_path = "models/sd_v2_base-57526ee4.ckpt"
    output_path = "out"
    train_batch_size = 1
    epochs = 1
    image_size = 512
    if os.path.exists(output_path):
        os.system(f"rm {output_path} -rf")
    if finetuning == "Vanilla":
        use_lora = False
    elif finetuning == "LoRA":
        use_lora = True
    cmd = (
        f"python train_text_to_image.py --data_path={data_path} --model_config={model_config_file} "
        f"--pretrained_model_path={pretrained_model_path} --weight_decay=0.01 --image_size={image_size} "
        f"--epochs={epochs} --ckpt_save_interval=1 --start_learning_rate=0.0001 --train_batch_size={train_batch_size} "
        f"--use_lora={use_lora} --output_path={output_path} --warmup_steps=0 --use_ema={use_ema} "
    )
    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # --------- Test running text_to_image.py using the trained model -----------
    end_ckpt = os.path.join(output_path, "ckpt", f"sd-{epochs}.ckpt")
    if finetuning == "Vanilla":
        use_lora = False
        lora_ckpt_path = None
        ckpt_path = end_ckpt
    elif finetuning == "LoRA":
        use_lora = True
        lora_ckpt_path = end_ckpt
        ckpt_path = "models/sd_v2_base-57526ee4.ckpt"
    cmd = (
        f"python text_to_image.py --config=configs/v2-inference.yaml --n_iter=1 --n_samples=2 "
        f"--output_path={output_path} --lora_ckpt_path={lora_ckpt_path} --use_lora={use_lora} "
        f"--ckpt_path={ckpt_path} --negative_prompt='cats' "
    )
    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "run text_to_image.py fails"


@pytest.mark.parametrize("use_ema", [True])
def test_train_infer_DreamBooth(use_ema):
    data_path = "data/Canidae/val/wolves"
    model_config_file = "configs/train_dreambooth_sd_v2.yaml"
    instance_prompt = "wolves"
    output_path = "out"
    pretrained_model_path = "models/sd_v2_base-57526ee4.ckpt"
    train_batch_size = 1
    epochs = 1
    image_size = 512
    if os.path.exists(output_path):
        os.system(f"rm {output_path} -rf")
    cmd = (
        f"python train_dreambooth.py --mode=0 --instance_data_dir={data_path} --instance_prompt='{instance_prompt}' "
        f"--model_config={model_config_file} --class_data_dir={data_path} --class_prompt='{instance_prompt}' "
        f"--pretrained_model_path={pretrained_model_path} --warmup_steps=200 --image_size={image_size} "
        f"--epochs={epochs} --start_learning_rate=0.00002 --train_batch_size={train_batch_size} --random_crop=True"
        f"--num_class_images=2 --output_path={output_path} --use_ema={use_ema}  --train_text_encoder=True "
        f"--train_data_repeats=2"
    )
    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # --------- Test running text_to_image.py using the trained model -----------
    end_ckpt = os.path.join(output_path, "ckpt/rank_0", f"sd-{epochs}.ckpt")
    cmd = (
        f"python text_to_image.py --config=configs/train_dreambooth_sd_v2.yaml --n_iter=1 --n_samples=2 "
        f"--output_path={output_path} --ckpt_path={end_ckpt} --negative_prompt='cats' "
    )
    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "run text_to_image.py fails"


if __name__ == "__main__":
    test_train_infer(use_ema=True, finetuning="LoRA")
    # test_train_infer_DreamBooth(use_ema=True)
