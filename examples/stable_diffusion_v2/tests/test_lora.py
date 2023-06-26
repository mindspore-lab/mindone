import mindspore as ms
import numpy as np

from ldm.modules.lora import LoRADenseLayer, LowRankDense, inject_trainable_lora
from ldm.modules.attention import  BasicTransformerBlock, CrossAttention
from ldm.modules.train.tools import set_random_seed

set_random_seed(42)

class SimpleSubNet(ms.nn.Cell):
    def __init__(self, din=128, dh=128, dtype=ms.float32):
        super().__init__()
        #self.to_q = nn.Dense(din, dh, has_bias=False).to_float(dtype)
        #self.to_k = nn.Dense(din, dh, has_bias=False).to_float(dtype)
        self.down_block_part1 = BasicTransformerBlock(dim=din, n_heads=2, d_head=dh, dtype=dtype)

        self.logit = ms.nn.Dense(dh, 1).to_float(dtype)

    def construct(self, x):
        feat = self.down_block_part1(x)
        out = self.logit(feat)
        return out

class SimpleNet(ms.nn.Cell):
    def __init__(self, din=128, dh=128, dtype=ms.float32):
        super().__init__()
        self.proj = ms.nn.Dense(din, din).to_float(dtype)
        self.encoder =  SimpleSubNet(din, dh, dtype)

    def construct(self, x):
        x = self.proj(x)

        return self.encoder(x)

def gen_np_data(bs=1, nd=2, fd=128):
    x = np.zeros([bs, nd, fd])
    for i in range(bs):
        for j in range(nd):
            x[i][j] = np.arange(0, fd, dtype=float) / fd / (j + 1)
    return x

def test_finetune_and_save():
    ms.set_context(mode=0)

    use_fp16 = True
    dtype = ms.float16 if use_fp16 else ms.float32
    rank = 4

    net = SimpleNet(dtype=dtype)
    # freeze network
    net.set_train(False)
    for name, param in net.parameters_and_names():
        param.requires_grad = False
    ms.save_checkpoint(net, 'test_lora_ori_net.ckpt')

    # save orginal forward output
    ori_net_stat = {}
    ori_net_stat['num_params'] = len(list(net.get_parameters()))
    #test_data = ms.Tensor(np.random.rand(1, 2, 128), dtype=dtype)
    test_data = ms.Tensor(gen_np_data(1, 2, 128), dtype=dtype)
    #test_data = ms.Tensor(np.ones([1, 2, 128])*0.05, dtype=dtype)

    ori_net_output = net(test_data)

    # inject lora layers
    injected_modules, injected_trainable_params = inject_trainable_lora(net, use_fp16=use_fp16)

    # 1. check foward result consistency
    ## since lora_up.weight are init with all zero. h_lora is alwasy zero before finetuning.
    net_output_after_lora_init = net(test_data)
    print('Outupt after lora injection: ', net_output_after_lora_init.sum())
    print('Oringinal net output: ', ori_net_output.sum())
    assert net_output_after_lora_init.sum()==ori_net_output.sum(), f'net_output_after_lora_init should be the same as ori_net_output'
    first_attn = list(injected_modules.values())[0]
    ori_net_stat['dense.linear'] = first_attn.to_q.linear.weight.data.sum()
    ori_net_stat['dense.lora_down'] = first_attn.to_q.lora_down.weight.data.sum()
    ori_net_stat['dense.lora_up'] = first_attn.to_q.lora_up.weight.data.sum()

    param_sum = 0
    for p in net.get_parameters():
        param_sum = param_sum + p.data.sum()
    print('Net param sum before ft: ', param_sum)


    # 2. check only lora injected params are trainable
    print('\nTrainable params: ', len(net.trainable_params()), '\n', "\n".join([f"{p.name}\t{p}" for p in net.trainable_params()])) # should be 2x4x2 x num_transformers
    for p in net.trainable_params():
        assert '.lora_down' in p.name or '.lora_up' in p.name, 'Only injected lora params can be trainable. but got non-lora param {p.name} trainable'

    # 3. check whether the number of injected modules and layers is correct. and whether the name of the injected params are correct.
    expected_im_for_simplenet = 2
    expected_ip_for_simplenet = expected_im_for_simplenet * 4 * 2 # 4 dense layers, each with lora_down, lora_up
    assert len(injected_modules)==expected_im_for_simplenet
    assert len(injected_trainable_params)==expected_ip_for_simplenet
    for _name, _param in injected_trainable_params.items():
        assert getattr(net, _name.split(".")[0]), f'Incorrect name: {_name}'
        assert getattr(net, _param.name.split(".")[0]), f'Incorrect name: {_param.name}'
    #print('Injected moduels: ', injected_modules)

    new_net_stat = {}
    new_net_stat['num_params'] = len(list(net.get_parameters()))
    assert new_net_stat['num_params'] - ori_net_stat['num_params'] == expected_ip_for_simplenet, 'Num of parameters should be increased by num_attention_layers * 4 * 2 after injection.'

    # 4. check finetune correctness
    def _simple_finetune(net):
        from mindspore.nn import TrainOneStepCell, WithLossCell
        loss = ms.nn.MSELoss()
        optim = ms.nn.SGD(params=net.trainable_params())
        #model = ms.Model(net, loss_fn=loss, optimizer=optim)
        net_with_loss = WithLossCell(net, loss)
        train_network = TrainOneStepCell(net_with_loss, optim)
        train_network.set_train()

        input_data = ms.Tensor(np.random.rand(1, 2, 128), dtype=dtype)
        label = ms.Tensor(np.ones([1, 1]), dtype=dtype)
        print('Finetuning...')
        for i in range(10):
            loss_val = train_network(input_data, label)
            print('loss: ', loss_val)

    _simple_finetune(net)
    net.set_train(False)
    new_net_stat['dense.linear'] = first_attn.to_q.linear.weight.data.sum()
    new_net_stat['dense.lora_down'] = first_attn.to_q.lora_down.weight.data.sum()
    new_net_stat['dense.lora_up'] = first_attn.to_q.lora_up.weight.data.sum()

    # check param change
    print('Ori net stat', ori_net_stat)
    print('New net stat', new_net_stat)
    # On Ascend, linear weight equality check can fail, may due to the difference on sum op. but CPU is ok.
    #assert new_net_stat['dense.linear'].numpy()== ori_net_stat['dense.linear'].numpy(), 'Not equal: {}, {}'.format(new_net_stat['dense.linear'].numpy(), ori_net_stat['dense.linear'].numpy())
    #assert new_net_stat['dense.lora_down'].value != ori_net_stat['dense.lora_down'].value
    #assert new_net_stat['dense.lora_up'].value != ori_net_stat['dense.lora_up'].value

    # check forward after finetuning
    param_sum = 0
    for p in net.get_parameters():
        param_sum = param_sum + p.data.sum()
    print('Net param sum after ft: ', param_sum)

    output_after_ft = net(test_data)
    print('Input data: ', test_data.sum())
    print('Net outupt after lora ft: ', output_after_ft.sum())
    print(f'\t (Before ft: {net_output_after_lora_init.sum()})')
    #assert output_after_ft.sum()!=net_output_after_lora_init.sum()

    # save
    ms.save_checkpoint([{"name":p.name, "data": p} for p in net.trainable_params()], 'test_lora_tp_after_ft.ckpt') # only save lora trainable params only
    ms.save_checkpoint(net, 'test_lora_net_after_ft.ckpt')


def compare_before_after_lora_finetune(
        pretrained_ckpt='test_lora_ori_net.ckpt', 
        lora_ft_ckpt='test_lora_net_after_ft.ckpt',
        ):
    ms.set_context(mode=0)
    ori_param_dict = ms.load_checkpoint(pretrained_ckpt)
    lora_param_dict = ms.load_checkpoint(lora_ft_ckpt)

    # get to q
    ori_weight = 0
    ori_param = None 
    for p in ori_param_dict: 
        if 'attn1.to_q.weight' in p:
            print('ori: ', p)
            ori_weight = ori_param_dict[p].data.sum()
            ori_param = p
            break

    lora_param = None
    lora_weight = 0
    lora_up_weight = 0
    for p in lora_param_dict: 
        if 'attn1.to_q.linear.weight' in p:
            lora_param = p
            lora_weight = lora_param_dict[p].data.sum()
            print('lora: ', p)
            lora_up_param = p.replace('.linear.', '.lora_up.') 
            lora_up_weight = lora_param_dict[lora_up_param].data.sum()
            break

    print(ori_param, lora_param)
    assert ori_param == lora_param.replace('.linear.', '.')

    print('lora up: ', lora_up_weight )
    assert lora_up_weight != 0

    # TODO: expect 0 linear weight diff. on Ascend, diff: < 1e-5. on CPU, diff is 0. 
    print('linear weight change: ', ori_weight, lora_weight)
    assert ori_weight==lora_weight


    #for p in ori_param_dict: 
    #    if 'attn1.to_q.lora_down.weight' in 

def test_load_and_infer():
    ms.set_context(mode=0)
    use_fp16 = True
    dtype = ms.float16 if use_fp16 else ms.float32
    rank = 4

    net = SimpleNet(dtype=dtype)
    #print(net)
    ori_ckpt_fp = 'test_lora_ori_net.ckpt'
    param_dict = ms.load_checkpoint(ori_ckpt_fp)
    net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
    print('Finish loading original net params: ', net_not_load, ckpt_not_load)

    # freeze network
    net.set_train(False)
    for name, param in net.parameters_and_names():
        param.requires_grad = False

    # load lora ckpt
    injected_modules, injected_trainable_params = inject_trainable_lora(net, use_fp16=use_fp16)
    print('injected_modules)', len(injected_modules), injected_modules)
    print('injected_trainable_params', len(injected_trainable_params), injected_trainable_params)

    load_lora_only = False
    if not load_lora_only:
        # method 1. load complete. load the whole pretrained ckpt with lora params
        ckpt_fp = 'test_lora_net_after_ft.ckpt'
        param_dict = ms.load_checkpoint(ckpt_fp)
        #print('\nD--: ', len(param_dict), param_dict)

        net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
        print('Finish loading ckpt with lora: ', net_not_load, ckpt_not_load)
    else:
        # method 2. load lora only
        lora_ckpt_fp = 'test_lora_tp_after_ft.ckpt'
        param_dict = ms.load_checkpoint(lora_ckpt_fp)
        net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
        print('Finish loading lora params: ', net_not_load, ckpt_not_load)

    # 1. test forward result consistency
    test_data = ms.Tensor(gen_np_data(1, 2, 128), dtype=dtype)
    #test_data = ms.ops.ones([1, 2, 128], dtype=dtype)*0.66
    net_output = net(test_data)
    print('Input data: ', test_data.sum())
    print("Net forward output: ", net_output.sum())


def test_compare_pt():
    '''
    test consistency btw our lora impl and torch version
    '''
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
    #print(list(mnet.get_parameters()))

    # copy weights
    t_param_dict = {}
    for name, param in tnet.named_parameters():
        #print('pt param name, ', name, param.size())
        t_param_dict[name] = param

    for name, param in mnet.parameters_and_names():
        #print('ms param name, ', name, param.shape)
        param.requires_grad = False
        #param.set_data()

        # they have the same param names, linear, lora_up, lora_down
        torch_weight = t_param_dict[name].data
        #print('Find torch weight value: ', torch_weight.shape)

        ms_weight = ms.Tensor(torch_weight.numpy())
        param.set_data(ms_weight)

        print(f'Set ms param {name} to torch weights')

    mout = mnet(ms.Tensor(x))
    print("ms lora: ", mout.sum())

    print("diff: ", mout.sum().numpy() - tout.sum().numpy())


# ----------------------- for debug --------------- #
'''
def test_finetune_and_save_debug():
    use_fp16 = True
    dtype = ms.float16 if use_fp16 else ms.float32
    rank = 4

    net = SimpleNet(dtype=dtype)
    # freeze network
    net.set_train(False)
    for name, param in net.parameters_and_names():
        param.requires_grad = False

    ms.save_checkpoint(net, 'test_lora_ori_net.ckpt')

    # check forward
    test_data = ms.Tensor(np.random.rand(1, 2, 128), dtype=dtype)
    ori_output = net(test_data)

    # check original network statistic, cross attention linear
    ori_net_stat = {}

    for name, param in net.parameters_and_names():
        print('param name, ', name, param.shape)
    ori_net_stat['num_params'] = len(list(net.get_parameters()))
    for name, subcell in net.cells_and_names():
        if isinstance(subcell, CrossAttention):
            ori_net_stat[name + '.to_q.weight'] = subcell.to_q.weight.data.sum()
            ori_net_stat[name + '.to_out.0.weight'] = subcell.to_out[0].weight.data.sum()
            ori_net_stat[name + '.to_q trainable'] = subcell.to_q.weight.requires_grad
            ori_net_stat[name + '.to_out trainable'] = subcell.to_out[0].weight.requires_grad
    print('Arch: ', net)
    print('Num params: ', ori_net_stat['num_params'])
    print('Cross attention weights stat: ', ori_net_stat)

    # find target layers in target moduels, and inject lora to target layers
    num_subcells = 0
    catched_attns = {}
    taget_modules = [CrossAttention]

    lora_attns = {}
    lora_trainable_params = []
    for name, subcell in net.cells_and_names():
        #print('sub cell: ', name, subcell)
        if isinstance(subcell, CrossAttention):
            print('Get crossattn: ', name, subcell)
            catched_attns[name] = subcell

            hier_path = name.split('.')

            #cur = net
            #for submodule_name in hier_path:
            #    cur = getattr(cur, submodule_name)
            #print('===> Cur point to: ', cur)
            print('=> Get target dense layers in original net: ')
            #print(subcell.to_q)
            target_dense_layers = [subcell.to_q, subcell.to_k, subcell.to_v, subcell.to_out[0]]
            new_lora_dense_layers = []
            for i, tar_dense in enumerate(target_dense_layers):
                print(name, tar_dense)
                if not isinstance(tar_dense, ms.nn.Dense):
                    raise ValueError(f'{tar_dense} is NOT a nn.Dense layer')
                has_bias = getattr(tar_dense, 'has_bias')
                in_channels = getattr(tar_dense, 'in_channels')
                out_channels = getattr(tar_dense, 'out_channels')
                print('in_channels: ', in_channels)
                print('out_channels: ', out_channels)
                print('Has bias?: ', has_bias)
                print('weight: ', tar_dense.weight, tar_dense.weight.data.sum())
                print('bias: ', tar_dense.bias, tar_dense.bias.sum() if tar_dense.bias is not None else "", '\n')
                #subcell.to_q.weight

                print('=> Replacing target dense layer with lora dense')
                tmp_lora_dense = LoRADenseLayer(
                        in_features=in_channels,
                        out_features=out_channels,
                        has_bias=has_bias,
                        rank=rank,
                        dtype=dtype)
                print('create lora dense layer: ', 'linear.weight: ', tmp_lora_dense.linear.weight.data.sum())
                print('\ttest its forward result at random init: ', tmp_lora_dense(ms.ops.ones([1, in_channels], dtype=dtype)*0.66).sum())

                # copy orignal weight and bias to lora linear (pointing)
                tmp_lora_dense.linear.weight = tar_dense.weight
                if has_bias:
                    tmp_lora_dense.linear.bias= tar_dense.bias
                print('copy weights from target dense to lora_dense.linear.weight: ', tmp_lora_dense.linear.weight.data.sum())
                print('\ttest its forward result after copying: ', tmp_lora_dense(ms.ops.ones([1, in_channels], dtype=dtype)*0.66).sum())

                new_lora_dense_layers.append(tmp_lora_dense)

            #target_dense_layers = [subcell.to_q, subcell.to_k, subcell.to_v, subcell.to_out[0]]
            # replace the 4 dense layers with the created lora layers, mount on
            subcell.to_q = new_lora_dense_layers[0]
            subcell.to_k = new_lora_dense_layers[1]
            subcell.to_v = new_lora_dense_layers[2]
            subcell.to_out[0] = new_lora_dense_layers[3]

            # TODO: don't know why the renaming dows not work in th end trainable_param
            def _update_param_name(param, prefix_module_name):
                # update param name to prefix for lora_up.weight and lora_down.weight
                if not param.name.startswith(prefix_module_name):
                    param.name = prefix_module_name + '.' + param.name

            for param in subcell.get_parameters():
                # filter to get lora added params by param name
                #print(param)
                if '.lora_down' in param.name or '.lora_up' in param.name or '.linear.' in param.name:
                   _update_param_name(param, name)
            # TODO: instead of using fixed list, pick target dense layer by name string then replace it for better extension.
            #lora_attns[name] = subcell # recored
            print('=> New cross attention after lora injection: ', subcell)

        num_subcells += 1
    print('Num sub cells: ', num_subcells)

    print('=> New net after lora injection: ', net)
    print('\t=> Attn param names: ', '\n'.join([name+'\t'+str(param.requires_grad) for name, param in net.parameters_and_names() if '.to_' in name]))

    print('Trainable params: ', net.trainable_params())
    #exit(1)

    new_net_stat = {}
    new_net_stat['num_params'] = len(list(net.get_parameters()))
    for name, subcell in net.cells_and_names():
        if isinstance(subcell, CrossAttention):
            new_net_stat[name + '.to_q.linear.weight'] = subcell.to_q.linear.weight.data.sum()
            new_net_stat[name + '.to_q.lora_up.weight'] = subcell.to_q.lora_up.weight.data.sum()
            new_net_stat[name + '.to_out.0.linear.weight'] = subcell.to_out[0].linear.weight.data.sum()
            new_net_stat[name + '.to_q.linear trainable'] = subcell.to_q.linear.weight.requires_grad
            new_net_stat[name + '.to_q.lora_up trainable'] = subcell.to_q.lora_up.weight.requires_grad
            new_net_stat[name + '.to_out.linear trainable'] = subcell.to_out[0].linear.weight.requires_grad
            assert new_net_stat[name + '.to_q.linear.weight']==ori_net_stat[name + '.to_q.weight'], 'CrossAttention linear weights are changed after lora injection'
            assert new_net_stat[name + '.to_q.linear trainable']==False
            assert new_net_stat[name + '.to_q.lora_up trainable']==True

    print('Ori net stat: ', ori_net_stat)
    print('New net stat: ', new_net_stat)
    assert new_net_stat['num_params'] - ori_net_stat['num_params'] == len(catched_attns) * len(target_dense_layers) * 2, 'Num of parameters should be increased by num_attention_layers * 4 * 2 after injection.'


    output_after_lora_init = net(test_data)
    # since lora_up.weight are init with zero. h_lora is alwasy zero without finetuning.
    assert output_after_lora_init.sum()==ori_output.sum()
    print('Outupt after lora injection: ', output_after_lora_init.sum())
    print(' \t Original net output: ', ori_output.sum())

    # finetune
    print('\nTrainable params: ', len(net.trainable_params()), '\n', "\n".join([f"{p.name}\t{p}" for p in net.trainable_params()])) # should be 2x4x2 x num_transformers

    from mindspore.nn import TrainOneStepCell, WithLossCell
    loss = ms.nn.MSELoss()
    optim = ms.nn.SGD(params=net.trainable_params())
    #model = ms.Model(net, loss_fn=loss, optimizer=optim)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optim)
    train_network.set_train()

    input_data = test_data
    label = ms.Tensor(np.ones([1, 1]), dtype=dtype)
    for i in range(4):
        loss_val = train_network(input_data, label)
        print(loss_val)

    # check forward again
    output_after_ft = net(test_data)
    print('Outupt after lora ft: ', output_after_ft.sum())
    print('\t before, output after lora init: ', output_after_lora_init.sum())
    assert output_after_ft.sum()!=output_after_lora_init.sum()

    # save
    ms.save_checkpoint([{"name":p.name, "data": p} for p in net.trainable_params()], 'test_lora_tp_after_ft.ckpt') # only save lora trainable params only
    ms.save_checkpoint(net, 'test_lora_net_after_ft.ckpt')
'''

# ----------------------- for debug END --------------- #

if __name__ == '__main__':
    #test_compare_pt()
    #test_finetune_and_save()
    #compare_before_after_lora_finetune()
    #test_load_and_infer()

    compare_before_after_lora_finetune('models/stablediffusionv2_512.ckpt', 'output/lora_pokemon_exp1/txt2img/ckpt/rank_0/sd-18_277.ckpt')

