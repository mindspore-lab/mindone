"""Modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py"""

import os
import time
from functools import partial

import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import Qwen2Config, Qwen2Tokenizer
from transformers.optimization import _get_cosine_schedule_with_warmup_lr_lambda

import mindspore as ms
import mindspore.mint as mint
from mindspore.dataset import GeneratorDataset
from mindspore.experimental.optim import AdamW, Optimizer
from mindspore.experimental.optim.lr_scheduler import LambdaLR

from mindone.trainers.muon import Muon
from mindone.transformers import Qwen2ForCausalLM


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class MoonDataset:
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.npy"):
            self.tokens = np.load(f"{self.dataset_name}.npy")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            np.save(f"{self.dataset_name}.npy", self.tokens)

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = np.asarray(token_slice, dtype=np.int32)
        return data


def get_model_and_dataloader(model_name, dataset_name, hidden_size):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    train_dataset = load_dataset(name2path[dataset_name])
    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer)
    # mike: default shuffle = True, for comparison set it to be False
    train_loader = GeneratorDataset(train_dataset, column_names="input_ids", shuffle=True).batch(8)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=513,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1, clip_value=None):
    if optimizer_name == "adamw":
        return AdamW(model.get_parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.parameters_and_names()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.parameters_and_names()
            if not (p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name)
        ]

        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
            clip_value=clip_value,
        )
    else:
        assert 0, "optimizer not supported"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--clip_value", type=float, default=None)
    args = parser.parse_args()
    logger.add(f"logs/train_{args.model}_{args.optimizer}_lr{args.lr}.log")

    ms.set_seed(0)
    model, train_loader = get_model_and_dataloader(args.model, args.dataset, args.hidden_size)
    optimizer = get_optimizer(args.optimizer, model, lr=args.lr, clip_value=args.clip_value)

    model.set_train(True)
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )

    total_train_params = sum([x.numel() for x in optimizer.parameters])
    logger.info(f"Total number of trainable parameters: {total_train_params:,}")

    grad_fn = ms.value_and_grad(model, grad_position=None, weights=optimizer.parameters, has_aux=True)
    for epoch in range(epoch):
        for step, batch in enumerate(train_loader.create_tuple_iterator()):
            (input_ids,) = batch
            (loss, _, qk_products), grads = grad_fn(input_ids=input_ids, labels=input_ids, return_dict=False)
            qk_products_max = max([mint.max(x).item() for x in qk_products])
            logger.info(f"QK max value: {qk_products_max:.3f}")
            ms.synchronize()
            start = time.time()
            optimizer(grads, qk_products)
            ms.synchronize()
            duration = time.time() - start
            lr_scheduler.step()
            logger.info(
                f"Epoch: {epoch} Step: {step} LR: {optimizer.param_groups[0]['lr'].item():.5f} "
                f"Optimizer update time: {duration:.3f} Training loss: {loss.item()}"
            )
