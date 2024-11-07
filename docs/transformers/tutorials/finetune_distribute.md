# Distributed training with mixed precision and ZeRO parallelism

The Trainer supports distributed training and mixed precision, which means you can also use it in a script. To enable both of these features:

See `examples/transformers/llama/finetune_with_mindspore_trainer.py` for more detail.

- Add the `is_distribute` argument to enable distribute training.
- Add the `fp16` or `bf16` argument to enable mixed precision.
- Add the `zero_stage` argument to enable optimizer parallelism with `ZeRO` algorithm.
- Set the number of global/local NPUs to use with the `worker_num`/`local_worker_num` argument.

```shell
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=outputs/parallel_logs \
python finetune_with_mindspore_trainer.py \
  --model_path $local_path/meta-llama/Meta-Llama-3-8B \
  --dataset_path $local_path/yelp_review_full \
  --output_dir ./outputs \
  --bf16 \
  --zero_stage 2 \
  --is_distribute True
```

Another example implemented through native MindSpore, see `examples/transformers/llama/finetune_in_native_mindspore.py` for more detail.

<details onclose>

```shell
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=outputs/parallel_logs \
python finetune_in_native_mindspore.py \
  --model_path meta-llama/Meta-Llama-3-8B \
  --dataset_path Yelp/yelp_review_full \
  --bf16 \
  --zero_stage 2 \
  --is_distribute True
```

</details>
