import argparse
from typing import Tuple

import numpy as np
from emu3.acceleration import create_parallel_group, get_sequence_parallel_group
from emu.mllm import Emu3ForCausalLM
from emu3.mllm.configuration_emu3 import Emu3Config

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.communication import get_group_size, init


class MeanNet(nn.Cell):
    def __init__(self, net: nn.Cell) -> None:
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        output = self.net(*inputs)[1]
        return output.mean() * 1024.0


def get_sample_data(dtype: ms.Type = ms.float32) -> Tuple[Tensor, ...]:
    seq_len = 10
    vocab_size = 184622
    input_ids = ops.randint(0, vocab_size, [1, seq_len], dtype=ms.int23)
    attention_mask = ops.ones_like(input_ids, dtype=ms.int23)
    labels = None
    # labels = input_ids.copy() # test training pipeline
    return input_ids, attention_mask, labels


def get_network_config():
    config = dict(num_hidden_layers=1, attn_implementation="eager", post_init_weight=False)
    config = {
        "architectures": ["Emu3ForCausalLM"],
        "attention_dropout": 0.1,
        "auto_map": {
            "AutoConfig": "configuration_emu3.Emu3Config",
            "AutoModelForCausalLM": "modeling_emu3.Emu3ForCausalLM",
        },
        "boi_token_id": 151852,
        "bos_token_id": 151849,
        "eof_token_id": 151847,
        "eoi_token_id": 151853,
        "eol_token_id": 151846,
        "eos_token_id": 151850,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "image_area": 262144,
        "img_token_id": 151851,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 5120,
        "model_type": "Emu3",
        "num_attention_heads": 32,
        "num_hidden_layers": 1, # DEBUG
        "num_key_value_heads": 8,
        "pad_token_id": 151643,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.44.0",
        "use_cache": False,
        "vocab_size": 184622,
        "attn_implementation": "flash_attention_2",
        "use_return_dict": False
    }
    return config


def run_network(mode: int = 0, dtype: ms.Type = ms.float32):
    ms.set_context(mode=mode)
    init()

    # prepare data
    ms.set_seed(1024)
    data = get_sample_data(dtype=dtype)

    run_parallel_network(data, dtype=dtype)


def run_parallel_network(data: Tuple[Tensor, ...], dtype: ms.Type = ms.float32):
    # non parallel network
    ms.set_seed(1024)
    non_parallel_network_cfg = get_network_config()
    config = Emu3Config(**non_parallel_network_cfg)
    non_parallel_network = Emu3ForCausalLM(config).set_train(False)
    # if data[-1] is not None:
    #     non_parallel_network.set_train(True)

    # parallel netowrk
    ms.set_seed(1024)
    create_parallel_group(shards=get_group_size())
    parallel_network_cfg = get_network_config()
    config = Emu3Config(**non_parallel_network_cfg)
    non_parallel_network = Emu3ForCausalLM(config).set_train(False)
    # if data[-1] is None:
    #     non_parallel_network = non_parallel_network.set_train(True)

    # load weight
    for (_, w0), (_, w1) in zip(non_parallel_network.parameters_and_names(), parallel_network.parameters_and_names()):
        w1.set_data(w0)
        np.testing.assert_allclose(w0.value().asnumpy(), w1.value().asnumpy())

    # test forward
    non_parallel_out = non_parallel_network(*data)
    # non_parallel_out_loss = non_parallel_out[0].asnumpy() if non_parallel_out[0] is not None else None
    non_parallel_out_logits = non_parallel_out[1].asnumpy()

    parallel_out = parallel_network(*data)
    # parallel_out_loss = parallel_out[0].asnumpy() if parallel_out[0] is not None else None
    parallel_out_logits = parallel_out[1].asnumpy()


    assert np.count_nonzero(non_parallel_out_logits) > 0
    np.testing.assert_equal(non_parallel_out_logits.shape, parallel_out_logits.shape)
    np.testing.assert_allclose(non_parallel_out_logits, parallel_out_logits, rtol=1.3e-6, atol=1e-5)
    print("Test 1 (Forward): Passed.", flush=True)

    # test backward
    non_parallel_mean_net = MeanNet(non_parallel_network)
    parallel_mean_net = MeanNet(parallel_network)

    # check the parameter gradient
    grad_fn = ms.grad(non_parallel_mean_net, grad_position=None, weights=non_parallel_mean_net.trainable_params())
    non_parallel_grads = grad_fn(*data)

    grad_fn = ms.grad(parallel_mean_net, grad_position=None, weights=parallel_mean_net.trainable_params())
    parallel_grads = grad_fn(*data)

    # take mean around different ranks
    sp_group = get_sequence_parallel_group()
    reduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=sp_group)
    num = get_group_size()
    syn_parallel_grads = list()
    for x in parallel_grads:
        syn_parallel_grads.append(reduce(x) / num)

    pass_grads = []
    for grad_0, grad_1 in zip(non_parallel_grads, syn_parallel_grads):
        is_passed = np.allclose(grad_0.asnumpy(), grad_1.asnumpy(), rtol=1.3e-6, atol=1e-5)
        pass_grads.append(is_passed)
    assert all(pass_grads), f"Pass rate ({sum(pass_grads)/len(pass_grads) * 100:.3f} %) is not 100 %"

    print("Test 2 (Backward: Parameter Gradient): Passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_network(mode=args.mode)
