# Wan2.2 LoRA Finetune

We provide an example of how to finetune Wan2.2 model (5B) Text-to-Video task using LoRA (Low-Rank Adaptation) technique.

## Prerequisites

Before running the finetuning script, ensure you have the following prerequisites:

#### Requirements
| mindspore |	ascend driver | firmware    | cann toolkit/kernel|
| :-------: | :-----------: | :---------: | :----------------: |
| 2.7.0     |  25.2.0       | 7.7.0.6.236 | 8.2.RC1            |

#### Dataset Preparation

Prepare your dataset using the format provided in [Disney-VideoGeneration-Dataset](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset).

## Start Finetuning

We use the script `scripts/train_lora_2p.sh` to start the finetuning process. You can modify the parameters in the script as needed.

```bash
bash scripts/train_lora_2p.sh
```

The lora checkpoint and the visualization results will be saved in the `output` directory by default.

## Fintune Result

After finetuning on 2 Ascend devices, we obtained the following results (Using `Disney-VideoGeneration-Dataset` Dataset):

Training loss curve:

![result](https://github.com/user-attachments/assets/e5a54d00-8d27-42d9-84f5-2adda7723b8d)


You can run inference with the fine‑tuned LoRA weights by adding `--lora_dir /path/to/lora_ckpt_dir` to the arguments of `generation.py`.

Here are some samples generated using the finetuned LoRA weights with prompt: *Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.*

| step 0        | step 100     |
|---------------|--------------|
| <video src='https://github.com/user-attachments/assets/30295d93-d447-4af6-85a4-3e483c4fa0ae' width=180/>   | <video src='https://github.com/user-attachments/assets/06cede81-fde8-4931-8abd-a5062e9dc74f' width=180/>               |
| step 200      | step 300     |
| <video src='https://github.com/user-attachments/assets/3754b2e5-25c5-4b41-8435-37e0f80c2753' width=180/>   |  <video src='https://github.com/user-attachments/assets/b3fc40c0-896d-4cf0-aa14-ae6a5597c19c' width=180/>             |



## Performance

|model         | precision | task          | resolution | card | batch size | recompute | s/step |
|--------------|-----------|---------------|------------|------| ---------- | --------- |--------|
|Wan2.2-TI2V-5B| bf16      | Text-To-Video | 704x1280   | 2    | 2          | ON        | 27     |
