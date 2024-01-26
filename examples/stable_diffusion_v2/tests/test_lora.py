import sys

import numpy as np

sys.path.append("./")
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.encoders.text_encoder import Transformer
from ldm.modules.lora import inject_trainable_lora_to_textencoder
from ldm.modules.train.tools import set_random_seed

sys.path.append("../../")
import mindspore as ms
from mindspore import ops

from mindone.models.lora import LoRADenseLayer, inject_trainable_lora, merge_lora_to_model_weights

set_random_seed(42)

# test_mode = "both"  # 'text_encoder' # ['text_encoder', 'unet', 'both']
test_mode = "unet"


class SimpleSubNet(ms.nn.Cell):
    def __init__(self, din=128, dh=128, dtype=ms.float32):
        super().__init__()
        # self.to_q = nn.Dense(din, dh, has_bias=False).to_float(dtype)
        # self.to_k = nn.Dense(din, dh, has_bias=False).to_float(dtype)
        self.down_block_part1 = BasicTransformerBlock(dim=din, n_heads=2, d_head=dh, dtype=dtype)

        self.logit = ms.nn.Dense(dh, 1).to_float(dtype)

    def construct(self, x):
        feat = self.down_block_part1(x)
        out = self.logit(feat)
        return out


class SimpleTextEnc(ms.nn.Cell):
    def __init__(self, din=128, dtype=ms.float32):
        super().__init__()
        self.enc = Transformer(
            width=din,
            layers=2,
            heads=2,
            attn_mask=self.build_attention_mask(77),
            dtype=dtype,
            epsilon=1e-6,
            use_quick_gelu=False,
        )

    @staticmethod
    def build_attention_mask(context_length):
        mask = np.triu(np.full((context_length, context_length), -np.inf).astype(np.float32), 1)
        return mask

    def construct(self, x):
        # x shape: (77, bs, dim)
        feat = self.enc(x)
        return feat


class SimpleNet(ms.nn.Cell):
    def __init__(self, din=128, dh=128, dtype=ms.float32):
        super().__init__()
        self.proj = ms.nn.Dense(din, din).to_float(dtype)
        self.encoder = SimpleSubNet(din, dh, dtype)

        self.text_enc = SimpleTextEnc(din=din, dtype=dtype)

    def construct(self, x):
        x1 = x[0]
        x2 = x[1]
        x1 = self.proj(x1)
        y1 = self.encoder(x1)

        y2 = self.text_enc(x2)
        return ops.mean(y1) + ops.mean(y2)


def gen_text_data(bs=1, cl=77, fd=128):
    x = np.zeros([bs, cl, fd])
    for i in range(bs):
        for j in range(cl):
            x[i][j] = np.arange(0, fd, dtype=float) / fd / (j + 1)

    x = np.transpose(x, (1, 0, 2))
    return x


def gen_data(bs=1, nd=2, fd=128, dtype=ms.float32):
    x = np.zeros([bs, nd, fd])
    for i in range(bs):
        for j in range(nd):
            x[i][j] = np.arange(0, fd, dtype=float) / fd / (j + 1)

    x_txt = gen_text_data(bs=bs, fd=fd)
    return (ms.Tensor(x, dtype=dtype), ms.Tensor(x_txt, dtype=dtype))


def test_finetune_and_save():
    ms.set_context(mode=1)

    use_fp16 = True
    dtype = ms.float16 if use_fp16 else ms.float32
    # rank = 4

    net = SimpleNet(dtype=dtype)

    # freeze network
    net.set_train(False)
    for name, param in net.parameters_and_names():
        param.requires_grad = False
    ms.save_checkpoint(net, "test_lora_ori_net.ckpt")

    # save orginal forward output
    ori_net_stat = {}
    ori_net_stat["num_params"] = len(list(net.get_parameters()))
    # test_data = ms.Tensor(np.random.rand(1, 2, 128), dtype=dtype)
    test_data = gen_data(1, 2, 128, dtype=dtype)
    # test_data = ms.Tensor(np.ones([1, 2, 128])*0.05, dtype=dtype)

    ori_net_output = net(test_data)

    # inject lora layers
    # if test_text_encoder:
    if test_mode in ["text_encoder", "both"]:
        injected_modules, injected_trainable_params = inject_trainable_lora_to_textencoder(net, use_fp16=use_fp16)
    if test_mode in ["unet", "both"]:
        injected_modules, injected_trainable_params = inject_trainable_lora(net, use_fp16=use_fp16)
    injected_trainable_tot = len(net.trainable_params())

    # 1. check forward result consistency
    # since lora_up.weight are init with all zero. h_lora is alwasy zero before finetuning.
    net_output_after_lora_init = net(test_data)
    print("Outupt after lora injection: ", net_output_after_lora_init.sum())
    print("Oringinal net output: ", ori_net_output.sum())
    assert (
        net_output_after_lora_init.sum() == ori_net_output.sum()
    ), "net_output_after_lora_init should be the same as ori_net_output"
    first_attn = list(injected_modules.values())[0]
    # if test_text_encoder:
    # ori_net_stat["dense.linear"] = first_attn.in_proj.linear.weight.data.sum()
    # ori_net_stat["dense.lora_down"] = first_attn.in_proj.lora_down.weight.data.sum()
    # ori_net_stat["dense.lora_up"] = first_attn.in_proj.lora_up.weight.data.sum()
    # else:
    ori_net_stat["dense.linear"] = first_attn.to_q.linear.weight.data.sum()
    ori_net_stat["dense.lora_down"] = first_attn.to_q.lora_down.weight.data.sum()
    ori_net_stat["dense.lora_up"] = first_attn.to_q.lora_up.weight.data.sum()

    param_sum = 0
    for p in net.get_parameters():
        param_sum = param_sum + p.data.sum()
    print("Net param sum before ft: ", param_sum)

    # 2. check only lora injected params are trainable
    print(
        "\nTrainable params: ",
        len(net.trainable_params()),
        "\n",
        "\n".join([f"{p.name}\t{p}" for p in net.trainable_params()]),
    )  # should be 2x4x2 x num_transformers
    for p in net.trainable_params():
        assert (
            ".lora_down" in p.name or ".lora_up" in p.name
        ), "Only injected lora params can be trainable. but got non-lora param {p.name} trainable"

    # 3. check whether the number of injected modules and layers is correct. and whether the name of the injected params
    # are correct.
    # num_dense_layers_in_attention = 4 if not test_text_encoder else 2
    expected_ip_unet = 2 * 4 * 2
    expected_ip_textenc = 2 * 2 * 2
    # expected_ip_for_simplenet = expected_im_for_simplenet * num_dense_layers_in_attention * 2  # 4 dense layers, each with lora_down, lora_up
    if test_mode == "both":
        expected_ip = expected_ip_unet + expected_ip_textenc
    elif test_mode == "unet":
        expected_ip = expected_ip_unet
    else:
        expected_ip = expected_ip_textenc
    assert injected_trainable_tot == expected_ip, "injected_trainable_params) == expected_ip"
    for _name, _param in injected_trainable_params.items():
        assert getattr(net, _name.split(".")[0]), f"Incorrect name: {_name}"
        assert getattr(net, _param.name.split(".")[0]), f"Incorrect name: {_param.name}"
    # print('Injected moduels: ', injected_modules)

    new_net_stat = {}
    new_net_stat["num_params"] = len(list(net.get_parameters()))
    assert (
        new_net_stat["num_params"] - ori_net_stat["num_params"] == expected_ip
    ), "Num of parameters should be increased by num_attention_layers * 4 * 2 after injection."

    # 4. check finetune correctness
    def _simple_finetune(net):
        from mindspore.nn import TrainOneStepCell, WithLossCell

        loss = ms.nn.MSELoss()
        optim = ms.nn.SGD(params=net.trainable_params())
        # model = ms.Model(net, loss_fn=loss, optimizer=optim)
        net_with_loss = WithLossCell(net, loss)
        train_network = TrainOneStepCell(net_with_loss, optim)
        train_network.set_train()

        # input_data = ms.Tensor(np.random.rand(1, 2, 128), dtype=dtype)
        train_data = gen_data(1, 2, 128, dtype=dtype)
        label = ms.Tensor(1.0, dtype=dtype)  # ms.Tensor(np.ones([1, 1]), dtype=dtype)
        print("Finetuning...")
        for i in range(10):
            loss_val = train_network(train_data, label)
            print("loss: ", loss_val)

    _simple_finetune(net)
    net.set_train(False)
    # if test_text_encoder:
    #    new_net_stat["dense.linear"] = first_attn.in_proj.linear.weight.data.sum()
    #    new_net_stat["dense.lora_down"] = first_attn.in_proj.lora_down.weight.data.sum()
    #    new_net_stat["dense.lora_up"] = first_attn.in_proj.lora_up.weight.data.sum()
    # else:
    new_net_stat["dense.linear"] = first_attn.to_q.linear.weight.data.sum()
    new_net_stat["dense.lora_down"] = first_attn.to_q.lora_down.weight.data.sum()
    new_net_stat["dense.lora_up"] = first_attn.to_q.lora_up.weight.data.sum()

    # check param change
    print("Ori net stat", ori_net_stat)
    print("New net stat", new_net_stat)
    # On Ascend, linear weight equality check can fail, may due to the difference on sum op. but CPU is ok.
    # assert new_net_stat['dense.linear'].numpy()== ori_net_stat['dense.linear'].numpy(),
    # 'Not equal: {}, {}'.format(new_net_stat['dense.linear'].numpy(), ori_net_stat['dense.linear'].numpy())
    # assert new_net_stat['dense.lora_down'].value != ori_net_stat['dense.lora_down'].value
    # assert new_net_stat['dense.lora_up'].value != ori_net_stat['dense.lora_up'].value

    # check forward after finetuning
    param_sum = 0
    for p in net.get_parameters():
        param_sum = param_sum + p.data.sum()
    print("Net param sum after ft: ", param_sum)

    output_after_ft = net(test_data)
    # print("Input data: ", test_data.sum())
    print("Net outupt after lora ft: ", output_after_ft.sum())
    print(f"\t (Before ft: {net_output_after_lora_init.sum()})")
    # assert output_after_ft.sum()!=net_output_after_lora_init.sum()

    # save
    ms.save_checkpoint(
        [{"name": p.name, "data": p} for p in net.trainable_params()], "test_lora_tp_after_ft.ckpt"
    )  # only save lora trainable params only
    ms.save_checkpoint(net, "test_lora_net_after_ft.ckpt")


def compare_before_after_lora_finetune(
    pretrained_ckpt="test_lora_ori_net.ckpt",
    lora_ft_ckpt="test_lora_net_after_ft.ckpt",
):
    ms.set_context(mode=0)
    ori_param_dict = ms.load_checkpoint(pretrained_ckpt)
    lora_param_dict = ms.load_checkpoint(lora_ft_ckpt)

    # get to q
    ori_weight = 0
    ori_param = None
    for p in ori_param_dict:
        if "attn1.to_q.weight" in p:
            print("ori: ", p)
            ori_weight = ori_param_dict[p].data.sum()
            ori_param = p
            break

    lora_param = None
    lora_weight = 0
    lora_up_weight = 0
    for p in lora_param_dict:
        if "attn1.to_q.linear.weight" in p:
            lora_param = p
            lora_weight = lora_param_dict[p].data.sum()
            print("lora: ", p)
            lora_up_param = p.replace(".linear.", ".lora_up.")
            lora_up_weight = lora_param_dict[lora_up_param].data.sum()
            break

    print(ori_param, lora_param)
    assert ori_param == lora_param.replace(".linear.", ".")

    print("lora up: ", lora_up_weight)
    assert lora_up_weight != 0

    # TODO: expect 0 linear weight diff. on Ascend, diff: < 1e-5. on CPU, diff is 0.
    print("linear weight change: ", ori_weight, lora_weight)
    assert ori_weight == lora_weight

    # for p in ori_param_dict:
    #    if 'attn1.to_q.lora_down.weight' in


def test_load_and_infer(merge_lora=False):
    ms.set_context(mode=0)
    use_fp16 = True
    dtype = ms.float16 if use_fp16 else ms.float32
    # rank = 4

    net = SimpleNet(dtype=dtype)
    # print(net)
    ori_ckpt_fp = "test_lora_ori_net.ckpt"
    param_dict = ms.load_checkpoint(ori_ckpt_fp)
    net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
    print("Finish loading original net params: ", net_not_load, ckpt_not_load)

    # freeze network
    net.set_train(False)
    for name, param in net.parameters_and_names():
        param.requires_grad = False

    # load lora ckpt
    if not merge_lora:
        if test_mode in ["text_encoder", "both"]:
            injected_modules, injected_trainable_params = inject_trainable_lora_to_textencoder(net, use_fp16=use_fp16)
        if test_mode in ["unet", "both"]:
            injected_modules, injected_trainable_params = inject_trainable_lora(net, use_fp16=use_fp16)
        # print("injected_modules)", len(injected_modules), injected_modules)
        # print("injected_trainable_params", len(injected_trainable_params), injected_trainable_params)

        load_lora_only = True
        if not load_lora_only:
            # method 1. load complete. load the whole pretrained ckpt with lora params
            ckpt_fp = "test_lora_net_after_ft.ckpt"
            param_dict = ms.load_checkpoint(ckpt_fp)
            # print('\nD--: ', len(param_dict), param_dict)

            net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
            print("Finish loading ckpt with lora: ", net_not_load, ckpt_not_load)
        else:
            # method 2. load lora only
            lora_ckpt_fp = "test_lora_tp_after_ft.ckpt"
            param_dict = ms.load_checkpoint(lora_ckpt_fp)
            net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
            print("Finish loading lora params: ", net_not_load, ckpt_not_load)
    else:
        lora_ckpt_fp = "test_lora_tp_after_ft.ckpt"
        net = merge_lora_to_model_weights(net, lora_ckpt_fp)

    # 1. test forward result consistency
    # test_data = ms.Tensor(gen_np_data(1, 2, 128), dtype=dtype)
    # test_data = ms.ops.ones([1, 2, 128], dtype=dtype)*0.66
    test_data = gen_data(1, 2, 128, dtype=dtype)
    net_output = net(test_data)
    # print("Input data: ", test_data.sum())
    print("Net forward output: ", net_output.sum())


def test_compare_pt():
    """
    test consistency btw our lora impl and torch version
    """
    from lora_torch import LoraInjectedLinear

    din = 128
    dout = 128
    r = 4
    x = np.random.rand(2, din).astype(np.float32)
    use_fp16 = True
    dtype = ms.float32
    if use_fp16:
        x = x.astype(np.float16)
        dtype = ms.float16

    # torch
    import torch

    with torch.no_grad():
        tnet = LoraInjectedLinear(din, dout, r=r)
        tout = tnet(torch.Tensor(x))
    print("torch lora: ", tout.sum())

    # ms
    mnet = LoRADenseLayer(din, dout, has_bias=False, rank=r, dtype=dtype)
    mnet.set_train(False)
    # print(list(mnet.get_parameters()))

    # copy weights
    t_param_dict = {}
    for name, param in tnet.named_parameters():
        # print('pt param name, ', name, param.size())
        t_param_dict[name] = param

    for name, param in mnet.parameters_and_names():
        # print('ms param name, ', name, param.shape)
        param.requires_grad = False
        # param.set_data()

        # they have the same param names, linear, lora_up, lora_down
        torch_weight = t_param_dict[name].data
        # print('Find torch weight value: ', torch_weight.shape)

        ms_weight = ms.Tensor(torch_weight.numpy())
        param.set_data(ms_weight)

        print(f"Set ms param {name} to torch weights")

    mout = mnet(ms.Tensor(x))
    print("ms lora: ", mout.sum())

    print("diff: ", mout.sum().numpy() - tout.sum().numpy())


if __name__ == "__main__":
    # test_compare_pt()
    test_finetune_and_save()
    # compare_before_after_lora_finetune()
    test_load_and_infer(merge_lora=True)

    # compare_before_after_lora_finetune(
    # 'models/sd_v2_base-57526ee4.ckpt', 'output/lora_pokemon_exp1/txt2img/ckpt/rank_0/sd-18_277.ckpt')
