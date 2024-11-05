import os
import argparse
import evaluate
import numpy as np
import mindspore as ms

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field

from mindone.transformers.models.llama import LlamaForSequenceClassification
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments
from mindone.transformers.mindspore_adapter import HF2MSDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="pretrained model name")
    parser.add_argument("--dataset_path", type=str, default="Yelp/yelp_review_full", help="dataset path.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="dataset path.")
    parser.add_argument("--rank_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    # 1. create dataset
    dataset = load_dataset(args.dataset_path)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,  # Note: pad is need for training batch size is gather than 1.
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"]

    model = LlamaForSequenceClassification.from_pretrained(args.model_path, num_labels=5)

    ds_init_params = {
        "num_parallel_workers": self.args.dataloader_num_workers,
        "sampler": self._get_train_sampler(),
        "python_multiprocessing": False,
        "num_shards": getattr(self.args, "rank_size", 1),
        "shard_id": getattr(self.args, "rank", 0),
        "column_names": "item"
    }

    train_dataloader = ms.dataset.GeneratorDataset(
        HF2MSDataset(small_train_dataset),
        num_parallel_workers=1,
        python_multiprocessing=False,
        num_shards=args.rank_size,
        shard_id=args.rank,
        column_names="item"
    )
    train_dataloader = train_dataloader.batch(
        batch_size=args.train_batch_size,
        num_parallel_workers=1,
        per_batch_map=lambda data_tuples: [np.stack([d[idx] for d in data_tuples], axis=0) for idx in range(4)],
        drop_remainder=True
    )
    train_dataloader = train_dataloader.repeat(1)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = ms.dataset.GeneratorDataset(
        eval_data,
        sampler=eval_sampler,
        num_parallel_workers=1,
        python_multiprocessing=False,
    )
    eval_dataloader = eval_dataloader.batch(
        batch_size=args.eval_batch_size,
        num_parallel_workers=1,
        per_batch_map=lambda data_tuples: [np.stack([d[idx] for d in data_tuples], axis=0) for idx in range(4)],
        drop_remainder=False
    )
    eval_dataloader = eval_dataloader.repeat(1)

    # Prepare optimizer
    if args.do_train:
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        lr_scheduler = get_linear_schedule_with_warmup(
            args.learning_rate, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        optimizer = nn.AdamWeightDecay(optimizer_grouped_parameters, learning_rate=lr_scheduler, eps=args.adam_epsilon)

        class NetWithLoss(nn.Cell):
            def __init__(self, model, lm_coef):
                super(NetWithLoss, self).__init__(auto_prefix=False)
                self.model = model
                self.lm_coef = lm_coef

            def construct(self, *args, **kwargs):
                losses = self.model(*args, **kwargs)
                return self.lm_coef * losses[0] + losses[1]

        train_step_fn = TrainOneStepWrapper(
            network=NetWithLoss(model, args.lm_coef),
            optimizer=optimizer,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad="global",
            clip_value=args.max_grad_norm
        )

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.set_train(True)
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            train_iterator = train_dataloader.create_tuple_iterator(num_epochs=1, output_numpy=True)
            for step, batch in enumerate(train_iterator):
                input_ids, mc_token_ids, lm_labels, mc_labels = batch

                # to tensor
                input_ids, mc_token_ids, lm_labels, mc_labels = \
                    ms.Tensor(input_ids), ms.Tensor(mc_token_ids), ms.Tensor(lm_labels), ms.Tensor(mc_labels)

                loss = train_step_fn(input_ids, mc_token_ids=mc_token_ids, lm_labels=lm_labels, mc_labels=mc_labels)
                tr_loss += loss.asnumpy().item()
                exp_average_loss = (
                    loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
                )
                nb_tr_steps += 1
                logger.info("Epoch: {}, Step: {}, Training loss: {:.2e}".format(epoch, step, exp_average_loss))

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model  # Only save the model itself

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        ms.save_checkpoint(model_to_save, output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

    if args.do_eval:
        model.set_train(False)
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_iterator = eval_dataloader.create_tuple_iterator(num_epochs=1, output_numpy=True)
        for batch in eval_iterator:
            input_ids, mc_token_ids, lm_labels, mc_labels = batch

            # to tensor
            input_ids, mc_token_ids, lm_labels, mc_labels = \
                ms.Tensor(input_ids), ms.Tensor(mc_token_ids), ms.Tensor(lm_labels), ms.Tensor(mc_labels)

            _, mc_loss, _, mc_logits = model(
                input_ids, mc_token_ids=mc_token_ids, lm_labels=lm_labels, mc_labels=mc_labels
            )

            mc_logits = mc_logits.asnumpy()
            mc_labels = mc_labels.asnumpy()
            tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

            eval_loss += mc_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.shape[0]
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        train_loss = tr_loss / nb_tr_steps if args.do_train else None
        result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy, "train_loss": train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == '__main__':
    main()
