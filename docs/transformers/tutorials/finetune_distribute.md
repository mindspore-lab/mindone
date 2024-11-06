# Distributed training and mixed precision


The Trainer supports distributed training and mixed precision, which means you can also use it in a script. To enable both of these features:

- Add the fp16 argument to enable mixed precision.
- Set the number of GPUs to use with the `worker_num` argument.

```shell
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=outputs/parallel_logs \
python finetune_with_mindspore_trainer.py \
  --model_path $local_path/meta-llama/Meta-Llama-3-8B \
  --dataset_path $local_path/yelp_review_full \
  --output_dir ./outputs \
  --per_device_train_batch_size 8 \
  --is_distribute True \
  --zero_stage 2 \
  --fp16
```