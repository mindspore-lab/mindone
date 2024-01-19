import logging

import ldm
from ldm.util import is_old_ms_version

import mindspore as ms
import mindspore.common.initializer as init
from mindspore import nn
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import get_activation
from mindspore.ops.primitive import Primitive

__all__ = [
    "LoRADenseLayer",
    "LowRankDense",
    "inject_trainable_lora",
    "inject_trainable_lora_to_textencoder",
    "freeze_non_lora_params",
    "get_lora_params",
]

_logger = logging.getLogger(__name__)


class LoRADenseLayer(nn.Cell):
    """
    Dense layer with lora injection, used to replace nn.Dense for lora fintuning.
    """

    def __init__(
        self,
        in_features,
        out_features,
        has_bias=True,
        rank=4,
        dropout_p=0.0,
        scale=1.0,
        dtype=ms.float32,
        activation=None,
    ):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")
        self.rank = rank
        self.scale = scale
        self.dtype = dtype

        # main/orginal linear layer
        self.linear = nn.Dense(in_features, out_features, has_bias=has_bias).to_float(dtype)

        # side-path/LoRA linear layers, the bias for lora matric should be False
        self.lora_down = nn.Dense(in_features, rank, has_bias=False).to_float(dtype)
        self.lora_up = nn.Dense(rank, out_features, has_bias=False).to_float(dtype)

        if is_old_ms_version():
            self.dropout = nn.Dropout(keep_prob=1 - dropout_p)
        else:
            self.dropout = nn.Dropout(p=dropout_p)

        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError(
                f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, but got "
                f"{type(activation).__name__}."
            )
        self.activation_flag = self.activation is not None

        self._init_weights()

    def _init_weights(self):
        self.lora_down.weight.set_data(
            init.initializer(
                init.Normal(sigma=1.0 / self.rank), self.lora_down.weight.shape, self.lora_down.weight.dtype
            )
        )
        self.lora_up.weight.set_data(
            init.initializer(init.Zero(), self.lora_up.weight.shape, self.lora_up.weight.dtype)
        )
        # note: no need to init linear layer since it will loaded by pretrained weights

    def construct(self, x):
        # ori_dtype = ops.dtype(x) # x.dtype
        # x = ops.cast(x, self.dtype)

        h_main = self.linear(x)

        z = self.lora_down(x)
        h_lora = self.lora_up(z)

        h = h_main + self.dropout(h_lora) * self.scale

        if self.activation_flag:
            h = self.activation(h)

        # h = ops.cast(h, ori_dtype)
        return h


def inject_trainable_lora_to_textencoder(
    net: nn.Cell, target_modules=["MultiheadAttention"], rank=4, dropout_p=0.0, scale=1.0, use_fp16=False, verbose=0
):
    """
    Search target moduels and layers in the network to inject LoRA trainable parameters.

    Note:
    1. Currently only support injection to dense layers in attention modules
    2. In order to find the target layers, currently the attention moduel must have the attribute of to_q, to_k,
        to_v and to_out[0], each of which correpsonds to a dense layer. to_out correspnds to a SquentialCell consisting
        of a dense layer and a dropout layer.
    """
    # TODO: due to qkv is not seperated, there are some difference from original impl.
    # though num added params are the same 3 x q_dim x rank, but q, k, v lora is influencing each other.
    target_modules = [getattr(ldm.modules.encoders.text_encoder, m) for m in target_modules]

    dtype = ms.float16 if use_fp16 else ms.float32
    ori_net_stat = {}
    ori_net_stat["num_params"] = len(list(net.get_parameters()))

    catched_attns = {}
    injected_modules = {}
    injected_trainable_params = {}

    # 1. search target modules
    for sc_name, subcell in net.cells_and_names():
        if isinstance(subcell, tuple(target_modules)):
            catched_attns[sc_name] = subcell
            # hier_path = name.split('.')
            # cur = net
            # for submodule_name in hier_path:
            #    cur = getattr(cur, submodule_name)

    if verbose:
        print("Found target modules for lora inject: ", catched_attns)

    if len(catched_attns) == 0:
        print(f"There is no target modules found in the network. Target modules {target_modules}")
        return net

    for sc_name, subcell in catched_attns.items():
        # 2. find target layers to be injected in the module
        # target_dense_layers = [subcell.to_q, subcell.to_k, subcell.to_v, subcell.to_out[0]]
        target_dense_layers = [subcell.in_proj, subcell.out_proj]
        # print(f'Target dense layers in the {sc_name}: ', target_dense_layers)

        # 3. create lora dense layers
        new_lora_dense_layers = []
        for i, tar_dense in enumerate(target_dense_layers):
            # print(name, tar_dense)
            if not isinstance(tar_dense, ms.nn.Dense):
                raise ValueError(f"{tar_dense} is NOT a nn.Dense layer")
            has_bias = getattr(tar_dense, "has_bias")
            in_channels = getattr(tar_dense, "in_channels")
            out_channels = getattr(tar_dense, "out_channels")

            if verbose:
                print(f"Create LoRA dense layer, of which linear weight is {tar_dense.weight.name}.")
            tmp_lora_dense = LoRADenseLayer(
                in_features=in_channels,
                out_features=out_channels,
                has_bias=has_bias,
                rank=rank,
                dtype=dtype,
                scale=scale,
            )

            # copy orignal weight and bias to lora linear (pointing)
            tmp_lora_dense.linear.weight = tar_dense.weight
            if has_bias:
                tmp_lora_dense.linear.bias = tar_dense.bias

            new_lora_dense_layers.append(tmp_lora_dense)

        # 4. replace target dense layers in attention module with the created lora layers and renaming the params
        # TODO:
        #  instead of using fixed list, pick target dense layer by name string then replace it for better extension.
        if verbose:
            print("Replacing target dense layers with the created lora layers.")
        subcell.in_proj = new_lora_dense_layers[0]
        subcell.out_proj = new_lora_dense_layers[1]

        def _update_param_name(param, prefix_module_name):
            if prefix_module_name not in param.name:
                param.name = prefix_module_name + "." + param.name

        for param in subcell.get_parameters():
            if ".lora_down" in param.name or ".lora_up" in param.name or ".linear." in param.name:
                _update_param_name(param, sc_name)

                if ".lora_down" in param.name or ".lora_up" in param.name:
                    injected_trainable_params[param.name] = param

    injected_modules = catched_attns

    if verbose:
        print(
            "Parameters in attention layers after lora injection: ",
            "\n".join([f"{p.name}\t{p}" for p in net.get_parameters() if "_proj" in p.name]),
        )
        # print('=> New net after lora injection: ', net)
        # print('\t=> Attn param names: ', '\n'.join([name+'\t'+str(param.requires_grad) for name,
        # param in net.parameters_and_names() if '.to_' in name]))

    new_net_stat = {}
    new_net_stat["num_params"] = len(list(net.get_parameters()))
    assert (
        new_net_stat["num_params"] - ori_net_stat["num_params"] == len(catched_attns) * len(target_dense_layers) * 2
    ), "Num of parameters should be increased by num_attention_layers * 2 * 2 after injection."
    assert len(injected_trainable_params) == len(injected_modules) * 2 * 2, (
        f"Expecting the number of injected lora trainable params to be {len(injected_modules)*2*2}, "
        f"but got {len(injected_trainable_params)}"
    )

    _logger.info(
        "LoRA enabled. Number of injected params: {}".format(new_net_stat["num_params"] - ori_net_stat["num_params"])
    )
    if verbose:
        print("Detailed injected params: \n", "\n".join([p.name + "\t" + f"{p}" for p in injected_trainable_params]))

    return injected_modules, injected_trainable_params


def inject_trainable_lora(
    net: nn.Cell,
    target_modules=["CrossAttention"],
    rank=4,
    dropout_p=0.0,
    scale=1.0,
    use_fp16=False,
    verbose=0,
):
    """
    Search target moduels and layers in the network to inject LoRA trainable parameters.

    Note:
    1. Currently only support injection to dense layers in attention modules
    2. In order to find the target layers, currently the attention moduel must have the attribute of to_q, to_k,
        to_v and to_out[0], each of which correpsonds to a dense layer. to_out correspnds to a SquentialCell consisting
        of a dense layer and a dropout layer.
    """
    target_modules = [getattr(ldm.modules.attention, m) for m in target_modules]

    dtype = ms.float16 if use_fp16 else ms.float32
    ori_net_stat = {}
    ori_net_stat["num_params"] = len(list(net.get_parameters()))

    catched_attns = {}
    injected_modules = {}
    injected_trainable_params = {}

    # 1. search target modules
    for sc_name, subcell in net.cells_and_names():
        if isinstance(subcell, tuple(target_modules)):
            catched_attns[sc_name] = subcell
            # hier_path = name.split('.')
            # cur = net
            # for submodule_name in hier_path:
            #    cur = getattr(cur, submodule_name)

    if verbose:
        print("Found target modules for lora inject: ", catched_attns)

    if len(catched_attns) == 0:
        print(f"There is no target modules found in the network. Target modules {target_modules}")
        return net

    for sc_name, subcell in catched_attns.items():
        # 2. find target layers to be injected in the module
        target_dense_layers = [subcell.to_q, subcell.to_k, subcell.to_v, subcell.to_out[0]]
        # print(f'Target dense layers in the {sc_name}: ', target_dense_layers)

        # 3. create lora dense layers
        new_lora_dense_layers = []
        for i, tar_dense in enumerate(target_dense_layers):
            # print(name, tar_dense)
            if not isinstance(tar_dense, ms.nn.Dense):
                raise ValueError(f"{tar_dense} is NOT a nn.Dense layer")
            has_bias = getattr(tar_dense, "has_bias")
            in_channels = getattr(tar_dense, "in_channels")
            out_channels = getattr(tar_dense, "out_channels")

            if verbose:
                print(f"Create LoRA dense layer, of which linear weight is {tar_dense.weight.name}.")
            tmp_lora_dense = LoRADenseLayer(
                in_features=in_channels,
                out_features=out_channels,
                has_bias=has_bias,
                rank=rank,
                dtype=dtype,
                scale=scale,
            )

            # copy orignal weight and bias to lora linear (pointing)
            tmp_lora_dense.linear.weight = tar_dense.weight
            if has_bias:
                tmp_lora_dense.linear.bias = tar_dense.bias

            new_lora_dense_layers.append(tmp_lora_dense)

        # 4. replace target dense layers in attention module with the created lora layers and renaming the params
        # TODO:
        #  instead of using fixed list, pick target dense layer by name string then replace it for better extension.
        if verbose:
            print("Replacing target dense layers with the created lora layers.")
        subcell.to_q = new_lora_dense_layers[0]
        subcell.to_k = new_lora_dense_layers[1]
        subcell.to_v = new_lora_dense_layers[2]
        subcell.to_out[0] = new_lora_dense_layers[3]

        def _update_param_name(param, prefix_module_name):
            if prefix_module_name not in param.name:
                param.name = prefix_module_name + "." + param.name

        for param in subcell.get_parameters():
            if ".lora_down" in param.name or ".lora_up" in param.name or ".linear." in param.name:
                _update_param_name(param, sc_name)

                if ".lora_down" in param.name or ".lora_up" in param.name:
                    injected_trainable_params[param.name] = param

    injected_modules = catched_attns

    if verbose:
        print(
            "Parameters in attention layers after lora injection: ",
            "\n".join([f"{p.name}\t{p}" for p in net.get_parameters() if "to_" in p.name]),
        )
        # print('=> New net after lora injection: ', net)
        # print('\t=> Attn param names: ', '\n'.join([name+'\t'+str(param.requires_grad) for name,
        # param in net.parameters_and_names() if '.to_' in name]))

    new_net_stat = {}
    new_net_stat["num_params"] = len(list(net.get_parameters()))
    assert (
        new_net_stat["num_params"] - ori_net_stat["num_params"] == len(catched_attns) * len(target_dense_layers) * 2
    ), "Num of parameters should be increased by num_attention_layers * 4 * 2 after injection."
    assert len(injected_trainable_params) == len(injected_modules) * 4 * 2, (
        f"Expecting the number of injected lora trainable params to be {len(injected_modules)*4*2}, "
        f"but got {len(injected_trainable_params)}"
    )

    _logger.info(
        "LoRA enabled. Number of injected params: {}".format(new_net_stat["num_params"] - ori_net_stat["num_params"])
    )
    if verbose:
        print("Detailed injected params: \n", "\n".join([p.name + "\t" + f"{p}" for p in injected_trainable_params]))

    return injected_modules, injected_trainable_params


def save_lora_trainable_params_only(net, ckpt_fp):
    ms.save_checkpoint(
        [{"name": p.name, "data": p} for p in net.trainable_params()], ckpt_fp
    )  # only save lora trainable params only


def load_lora_trainable_params_only(net, lora_ckpt_fp):
    """
    Load trained lora params to the network, which should have loaded pretrained params and injected with lora params.
    """
    # TODO: Cancel out the warning for not loading non-lora params.
    #  E.g. manually set target param values with lora params.
    param_dict = ms.load_checkpoint(lora_ckpt_fp)
    net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)


def get_lora_params(net, filter=None):
    injected_trainable_params = {}
    for param in net.get_parameters():
        if "lora_down." in param.name or "lora_up." in param.name:
            injected_trainable_params[param.name] = param

    return injected_trainable_params


# --- the code below is for architecture hacking-based implementation.
class LowRankDense(nn.Cell):
    """
    The lora side-path module with low rank matrices
    """

    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        dropout_p=0.0,
        scale=1.0,
        dtype=ms.float32,
    ):
        super().__init__()
        self.scale = scale
        self.lora_down = nn.Dense(in_features, rank, has_bias=False).to_float(dtype)
        self.lora_up = nn.Dense(rank, out_features, has_bias=False).to_float(dtype)
        if is_old_ms_version():
            self.dropout = nn.Dropout(keep_prob=1 - dropout_p)
        else:
            self.dropout = nn.Dropout(p=dropout_p)

        self.lora_down.weight.set_data(
            init.initializer(init.Normal(sigma=1.0 / rank), self.lora_down.weight.shape, self.lora_down.weight.dtype)
        )
        self.lora_up.weight.set_data(
            init.initializer(init.Zero(), self.lora_up.weight.shape, self.lora_up.weight.dtype)
        )

    def construct(self, x):
        z = self.lora_down(x)
        h_lora = self.lora_up(z)

        return self.dropout(h_lora) * self.scale


def freeze_non_lora_params(net, filter=None):
    injected_trainable_params = {}
    for param in net.get_parameters():
        if "lora_down." in param.name or "lora_up." in param.name:
            param.requires_grad = True
            injected_trainable_params[param.name] = param
        else:
            param.requires_grad = False

    return injected_trainable_params
