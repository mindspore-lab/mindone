import os, sys
import numpy as np
from PIL import Image
import mindspore as ms
from mindspore import amp
import torch
from easydict import EasyDict as edict

sys.path.insert(0, ".")

from hyvideo.modules.models import MMDoubleStreamBlock, MMSingleStreamBlock, HYVideoDiffusionTransformer
from hyvideo.modules.attention import VanillaAttention
from hyvideo.modules.token_refiner import SingleTokenRefiner


np.random.seed(42)

(B, L, N, D) = (1, 32, 6, 48)
L_txt = 18
hidden_size = N*D

img_shape = (B, L, N*D)
txt_shape = (B, L_txt, N*D)
vec_shape = (B, N*D)
freqs_cis_shape = (L, D)
img_ = np.random.normal(size=img_shape).astype(np.float32)
txt_ = np.random.normal(size=txt_shape).astype(np.float32)
vec_ = np.random.normal(size=vec_shape).astype(np.float32)
freqs_cis_cos_ = np.random.normal(size=freqs_cis_shape).astype(np.float32)
freqs_cis_sin_ = np.random.normal(size=freqs_cis_shape).astype(np.float32)


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)

def _convert_ckpt(pt_ckpt, rename_norm=False):
    # sd = torch.load(pt_ckpt, map_location="CPU")['model_state_dict']
    sd = torch.load(pt_ckpt)["model_state_dict"]
    target_data = []

    for k in sd:
        if rename_norm and ("norm" in k):
            ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
        else:
            ms_name = k
        target_data.append({"name": ms_name, "data": ms.Tensor(sd[k].detach().cpu().numpy())})

    save_fn = pt_ckpt.replace(".pth", ".ckpt")
    ms.save_checkpoint(target_data, save_fn)

    return save_fn




def test_attn():
    x_shape = (B, L, N, D) = (1, 32, 6, 48)
    q = np.random.normal(size=x_shape).astype(np.float32)
    k = np.random.normal(size=x_shape).astype(np.float32)
    v = np.random.normal(size=x_shape).astype(np.float32)
    q = ms.Tensor(q)
    k = ms.Tensor(k)
    v = ms.Tensor(v)

    # out = attention(q, k, v, mode='vanilla')

    # print(out.shape)


def test_dualstream_block():
    img = ms.Tensor(img_)
    txt = ms.Tensor(txt_)
    vec = ms.Tensor(vec_)
    freqs_cis = (ms.Tensor(freqs_cis_cos_), ms.Tensor(freqs_cis_sin_))
    # freqs_cis_cos = ms.Tensor(freqs_cis_cos)
    # freqs_cis_sin = ms.Tensor(freqs_cis_sin)

    print('input sum: ', img.sum())
    block = MMDoubleStreamBlock(
        hidden_size = hidden_size,
        heads_num=N,
        mlp_width_ratio=1,
        qkv_bias=True,
    )

    img_out, txt_out = block(img, txt, vec, freqs_cis=freqs_cis)
    # out = block(img, txt, vec, freqs_cis_cos=freqs_cis_cos, freqs_cis_sin=freqs_cis_sin)
    print(img_out.shape)
    print(img_out.mean(), img_out.std())  # -0.00013358534 1.0066756 for fp32
    print(txt.shape)
    print(txt_out.mean(), txt_out.std())


def test_singlestream_block(pt_fp: str=None):
    img = ms.Tensor(img_)
    txt = ms.Tensor(txt_)
    x = ms.ops.concat([img, txt], 1)
    vec = ms.Tensor(vec_)
    freqs_cis = (ms.Tensor(freqs_cis_cos_), ms.Tensor(freqs_cis_sin_))

    print('input sum: ', x.sum())
    block = MMSingleStreamBlock(
        hidden_size = hidden_size,
        heads_num=N,
        mlp_width_ratio=1,
    )

    out = block(x, vec, L_txt, freqs_cis=freqs_cis)

    print(out.shape)
    print(out.mean(), out.std())

    if pt_fp:
        pt_out = np.load(pt_fp)
        diff = _diff_res(out.asnumpy(), pt_out)
        print(diff)


def test_token_refiner(pt_fp=None):
    token_shape = (bs, max_text_len, emb_dim) = 1, 32, 64
    x = np.random.normal(size=token_shape).astype(np.float32)
    t = np.array([1000. for _ in range(bs)], dtype=np.float32)
    mask = np.zeros(shape=(bs, max_text_len), dtype=np.int32)
    mask[0, :4] = 1

    x = ms.Tensor(x)
    t = ms.Tensor(t)
    mask = ms.Tensor(mask)

    block = SingleTokenRefiner(
        in_channels=emb_dim,
        hidden_size=emb_dim,
        heads_num=8,
        depth=1,
        mlp_width_ratio=1,
    )

    ckpt = _convert_ckpt(f"tests/token_refiner.pth")
    sd = ms.load_checkpoint(ckpt)
    m, u = ms.load_param_into_net(block, sd)
    print("net param not loaded: ", m)
    print("ckpt param not loaded: ", u)

    out = block(x, t, mask)
    print(out.shape)
    print(out.mean(), out.std())

    if pt_fp:
        pt_out = np.load(pt_fp)
        diff = _diff_res(out.asnumpy(), pt_out)
        print(diff)


def test_hyvtransformer():
    token_shape = (bs, max_text_len, llm_emb_dim) = 1, 32, 64
    latent_shape = (bs, C, T, H, W) = (bs, 4, 5, 8, 8)
    clip_txt_len, clip_emb_dim = 18, 24
    patch_size = 2
    S_vid = T * (H//patch_size) * (W//patch_size)
    num_heads = 6
    hidden_size = 6*32
    pe_dim = head_dim = hidden_size // num_heads
    freqs_cos_shape = (S_vid, head_dim)

    video_latent = np.random.normal(size=latent_shape).astype(np.float32)
    t = np.array([1000. for _ in range(bs)], dtype=np.float32)
    text_states = np.random.normal(size=token_shape).astype(np.float32)
    text_mask = np.zeros(shape=(bs, max_text_len), dtype=np.int32)  #
    text_states_2 = np.random.normal(size=(bs, clip_emb_dim)).astype(np.float32) # [1, 768]
    freqs_cos = np.random.normal(size=freqs_cos_shape).astype(np.float32)
    freqs_sin = np.random.normal(size=freqs_cos_shape).astype(np.float32)
    guidance = np.array([7.0*1000 for _ in range(bs)], dtype=np.float32)
    text_mask[0, :4] = 1

    video_latent = ms.Tensor(video_latent)
    t = ms.Tensor(t)
    text_states = ms.Tensor(text_states)
    text_mask = ms.Tensor(text_mask)
    text_states_2 = ms.Tensor(text_states_2)
    freqs_cos = ms.Tensor(freqs_cos)
    freqs_sin = ms.Tensor(freqs_sin)
    guidance = ms.Tensor(guidance)

    # model
    args = edict()
    args.text_states_dim = llm_emb_dim
    args.text_states_dim_2 = clip_emb_dim
    args.model = 'HYVideo-T/2'
    DEBUG_CONFIG = {
        "HYVideo-T/2": {
            "mm_double_blocks_depth": 1,
            "mm_single_blocks_depth": 1,
            "rope_dim_list": [4, 14, 14], # [16, 56, 56], list sum = head_dim = pe_dim
            "hidden_size":  hidden_size,
            "heads_num": num_heads,
            "mlp_width_ratio": 1,
        },
    }
    factor_kwargs = {'dtype': ms.float32}

    block = HYVideoDiffusionTransformer(
            args,
            in_channels=C,
            **DEBUG_CONFIG[args.model],
            **factor_kwargs,
        )
    amp.auto_mixed_precision(block, amp_level='O2', dtype=ms.bfloat16)

    # run
    out = block(video_latent, t, text_states, text_mask, text_states_2, freqs_cos, freqs_sin, guidance)

    print(out.shape)
    print(out.mean(), out.std())


if __name__ == "__main__":
    # ms.set_context(mode=1)
    ms.set_context(mode=0, jit_syntax_level=ms.STRICT)
    # test_attn()
    # test_dualstream_block()
    # test_singlestream_block('tests/pt_single_stream.npy')
    # test_token_refiner('tests/pt_token_refiner.npy')
    test_hyvtransformer()

