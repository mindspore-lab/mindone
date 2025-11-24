# Wan2.2 LoRA Finetune

We provide an example of how to finetune Wan2.2 model (5B) Text-to-Video task using LoRA (Low-Rank Adaptation) technique.

## Prerequisites

Before running the finetuning script, ensure you have the following prerequisites:

#### Requirements
| mindspore |	ascend driver | firmware    | cann toolkit/kernel|
| :-------: | :-----------: | :---------: | :----------------: |
| 2.7.0     |  25.2.0       | 7.7.0.6.236 | 8.2.RC1            |

#### Dataset Preparation

Prepare your dataset following the format shown in `https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset`

## Start Finetuning

We use the script `scripts/train_lora_2p.sh` to start the finetuning process. You can modify the parameters in the script as needed.

```bash
bash scripts/train_lora_2p.sh
```

The lora checkpoint and the visualization results will be saved in the `output` directory by default.

## Fintune Result

After finetuning on 2 Ascend devices, we obtained the following results:

Training loss curve:



## Performance

|model         | precision | task          | resolution | card | batch size | recompute | s/step |
|--------------|-----------|---------------|------------|------| ---------- | --------- |--------|
|Wan2.2-TI2V-5B| bf16      | Text-To-Video | 704x1280   | 2    | 2          | ON        | 27     |
