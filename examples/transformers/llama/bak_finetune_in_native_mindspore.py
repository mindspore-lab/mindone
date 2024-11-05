#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Llama 3 model fine-tuning script.
    reference to https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
                 https://github.com/huggingface/transformers/blob/main/examples/legacy/run_openai_gpt.py

    This script with default values fine-tunes and evaluate a pretrained Meta Llama3 on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name meta-llama/Meta-Llama-3-8B \
          --do_train \
          --do_eval \
          --train_dataset "$ROC_STORIES_DIR/cloze_test_val__spring2016 - cloze_test_ALL_val.csv" \
          --eval_dataset "$ROC_STORIES_DIR/cloze_test_test__spring2016 - cloze_test_ALL_test.csv" \
          --output_dir ../log \
          --train_batch_size 16 \
"""

import argparse
import csv
import logging
import os
import random
import numpy as np
from tqdm import tqdm, trange

import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import GeneratorDataset
from mindspore.communication.management import init, get_rank, get_group_size

from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from mindone.transformers.models.llama import LlamaForCausalLM
from mindone.transformers.mindspore_adapter.utils import _is_parallel
from mindone.transformers.mindspore_adapter import (
    TrainOneStepWrapper,
    TensorDataset,
    RandomSampler,
    SequentialSampler
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def load_rocstories_dataset(dataset_path):
    """Output a list of tuples(story, 1st continuation, 2nd continuation, label)"""
    with open(dataset_path, encoding="utf_8") as f:
        f = csv.reader(f)
        output = []
        next(f)  # skip the first line
        for line in tqdm(f):
            output.append((" ".join(line[1:5]), line[5], line[6], int(line[-1]) - 1))
    return output


def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

    To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
    input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        lm_labels = np.full((n_batch, 2, input_len), fill_value=-100, dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for (
            i,
            (story, cont1, cont2, mc_label),
        ) in enumerate(dataset):
            with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
            with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
            input_ids[i, 0, : len(with_cont1)] = with_cont1
            input_ids[i, 1, : len(with_cont2)] = with_cont2
            mc_token_ids[i, 0] = len(with_cont1) - 1
            mc_token_ids[i, 1] = len(with_cont2) - 1
            lm_labels[i, 0, : len(with_cont1)] = with_cont1
            lm_labels[i, 1, : len(with_cont2)] = with_cont2
            mc_labels[i] = mc_label
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(all_inputs))
    return tensor_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="pretrained model name")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--train_dataset", type=str, default="")
    parser.add_argument("--eval_dataset", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=(
            "If > 0: set total number of training                         steps to perform. Override num_train_epochs."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before                        performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lm_coef", type=float, default=0.9)
    parser.add_argument("--n_valid", type=int, default=374)

    parser.add_argument("--rank_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    ms.set_seed(args.seed)

    n_npu = get_group_size() if _is_parallel() else 1
    logger.info("num cards: {}".format(n_npu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = ["_start_", "_delimiter_", "_classify_"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(special_tokens)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    model = LlamaForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Load and encode the datasets
    def tokenize_and_encode(obj):
        """Tokenize and encode a nested object"""
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return [tokenize_and_encode(o) for o in obj]

    logger.info("Encoding dataset...")
    train_dataset = load_rocstories_dataset(args.train_dataset)
    eval_dataset = load_rocstories_dataset(args.eval_dataset)
    datasets = (train_dataset, eval_dataset)
    encoded_datasets = tokenize_and_encode(datasets)

    # Compute the max input length for the Transformer
    max_length = model.config.n_positions // 2 - 2
    input_length = max(
        len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) + 3
        for dataset in encoded_datasets
        for story, cont1, cont2, _ in dataset
    )
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids)
    train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = ms.dataset.GeneratorDataset(
        train_data,
        sampler=train_sampler,
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
