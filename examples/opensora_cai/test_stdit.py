import os, sys
import torch
import mindspore as ms
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.models.stdit import STDiTBlock, STDiT_XL_2

ms.set_context(mode=1)

hidden_size = 1024
T = num_temporal = 1
S = num_spatial = 16*16  # num_patches // self.num_temporal

args = dict(
        hidden_size=hidden_size,
        num_heads=8,
        d_s=S,
        d_t=T,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    )

'''
x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
timestep (torch.Tensor): diffusion time steps; of shape [B]
y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
'''

B, C, T, H, W = 1, 4, 1, 32, 32
text_emb_dim = 4096
N_tokens = 64

x = np.random.normal(size=(B, C, T, H , W)).astype(np.float32)
t = np.random.randint(low=0, high=1000, size=B)
# condition, text, 
y = np.random.normal(size=(B, 1, N_tokens, text_emb_dim)).astype(np.float32)


global_inputs = (x, t, y)

def test_stdit():

    net = STDiT_XL_2()
    net.set_train(False)

    out = net(ms.Tensor(x), ms.Tensor(t), ms.Tensor(y), mask=None)
    print(out.shape)

def test_stdit_pt():
    pt_code_path = "/home/mindocr/yx/Open-Sora/"
    sys.path.append(pt_code_path)
    from opensora.models.stdit.stdit import STDiT_XL_2 as STD_PT

    net = STD_PT()
    net.eval()

    out = net(torch.Tensor(x), torch.Tensor(t), torch.Tensor(y), mask=None)
    print(out.shape)

if __name__ == "__main__":
    # ms.set_context(mode=1)
    # test_stdit()
    test_stdit_pt()



