# flake8: noqa

import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict as edict

import mindspore as ms
from mindspore import amp

sys.path.insert(0, ".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

# from hyvideo.modules.attention import VanillaAttention
from hyvideo.modules.models import (
    HUNYUAN_VIDEO_CONFIG,
    HYVideoDiffusionTransformer,
    MMDoubleStreamBlock,
    MMSingleStreamBlock,
)
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.modules.token_refiner import SingleTokenRefiner
from hyvideo.utils.helpers import set_model_param_dtype

np.random.seed(42)

(B, L, N, D) = (1, 32, 6, 48)
L_txt = 18
hidden_size = N * D

img_shape = (B, L, N * D)
txt_shape = (B, L_txt, N * D)
vec_shape = (B, N * D)
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
    state_dict = torch.load(pt_ckpt)  # , map_location=lambda storage, loc: storage)
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


def load_pt_checkpoint(model, ckpt_path, dtype=ms.float32, load_key="model_state_dict", remove_prefix=None):
    """
    model param dtype
    """
    import torch

    state_dict = torch.load(ckpt_path)
    sd = state_dict[load_key]
    parameter_dict = dict()

    for pname in sd:
        np_val = sd[pname].cpu().detach().float().numpy()
        if remove_prefix is not None and pname.startswith(remove_prefix):
            pname = pname[len(remove_prefix) :]
        else:
            pname = pname
        parameter_dict[pname] = ms.Parameter(ms.Tensor(np_val, dtype=dtype))

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, parameter_dict, strict_load=True)
    print("param not load: ", param_not_load)
    print("ckpt not load: ", ckpt_not_load)


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


def test_dualstream_block(pt_ckpt=None, pt_np=None, dtype=ms.float32):
    img = ms.Tensor(img_)
    txt = ms.Tensor(txt_)
    vec = ms.Tensor(vec_)
    freqs_cis = (ms.Tensor(freqs_cis_cos_), ms.Tensor(freqs_cis_sin_))
    # freqs_cis_cos = ms.Tensor(freqs_cis_cos)
    # freqs_cis_sin = ms.Tensor(freqs_cis_sin)

    print("input sum: ", img.sum())
    block = MMDoubleStreamBlock(
        hidden_size=3072,  # hidden size
        heads_num=24,
        mlp_width_ratio=4.0,
        qkv_bias=True,
        dtype=dtype,
        condition_type="token_replace",
    )
    if pt_ckpt:
        load_pt_checkpoint(block, pt_ckpt, load_key="module", remove_prefix="double_blocks.0.")

    if pt_np:
        if pt_np.endswith(".npy"):
            pt_dict = np.load(pt_np)
            pt_img_out, pt_txt_out = pt_dict["img_out"], pt_dict["txt_out"]
            img_diff = _diff_res(img_out.asnumpy(), pt_img_out)
            txt_diff = _diff_res(txt_out.asnumpy(), pt_txt_out)
            print("img diff: ", img_diff)
            print("txt diff: ", txt_diff)
        elif pt_np.endswith(".npz"):
            data = np.load(pt_np)
            img = ms.tensor(data["img"], dtype=dtype)
            txt = ms.tensor(data["txt"], dtype=dtype)
            vec = ms.tensor(data["vec"], dtype=dtype)
            freqs_cis = ms.tensor(data["freqs_cis"][0]), ms.Tensor(data["freqs_cis"][1])
            token_replace_vec = ms.tensor(data["token_replace_vec"], dtype=dtype)
            frist_frame_token_num = int(data["frist_frame_token_num"])
            actual_seq_qlen = ms.tensor(data["cu_seqlens_q"][1:], dtype=ms.int32)
            actual_seq_kvlen = ms.tensor(data["cu_seqlens_kv"][1:], dtype=ms.int32)
            img_out, txt_out = block(
                img,
                txt,
                vec,
                freqs_cos=freqs_cis[0],
                freqs_sin=freqs_cis[1],
                token_replace_vec=token_replace_vec,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                frist_frame_token_num=frist_frame_token_num,
            )
            pt_img_out, pt_txt_out = data["output_img"], data["output_txt"]
            img_diff = _diff_res(img_out.asnumpy(), pt_img_out)
            txt_diff = _diff_res(txt_out.asnumpy(), pt_txt_out)
            print("img diff: ", img_diff)
            print("txt diff: ", txt_diff)
    else:
        img_out, txt_out = block(img, txt, vec, freqs_cis=freqs_cis)
        # out = block(img, txt, vec, freqs_cis_cos=freqs_cis_cos, freqs_cis_sin=freqs_cis_sin)
        print(img_out.shape)
        print(img_out.mean(), img_out.std())
        print(txt.shape)
        print(txt_out.mean(), txt_out.std())


def test_singlestream_block(pt_ckpt: str = None, pt_np: str = None):
    img = ms.Tensor(img_)
    txt = ms.Tensor(txt_)
    x = ms.ops.concat([img, txt], 1)
    vec = ms.Tensor(vec_)
    freqs_cis = (ms.Tensor(freqs_cis_cos_), ms.Tensor(freqs_cis_sin_))

    print("input sum: ", x.sum())
    block = MMSingleStreamBlock(
        hidden_size=hidden_size,
        heads_num=N,
        mlp_width_ratio=1,
    )
    # load ckpt
    if pt_ckpt:
        load_pt_checkpoint(block, pt_ckpt)

    out = block(x, vec, L_txt, freqs_cis=freqs_cis)

    print(out.shape)
    print(out.mean(), out.std())

    if pt_np:
        pt_out = np.load(pt_np)
        diff = _diff_res(out.asnumpy(), pt_out)
        print(diff)


def test_token_refiner(pt_ckpt=None, pt_np=None, attn_mode="flash", dtype=ms.float32):
    token_shape = (bs, max_text_len, emb_dim) = 1, 32, 64
    x = np.random.normal(size=token_shape).astype(np.float32)
    t = np.array([1000.0 for _ in range(bs)], dtype=np.float32)
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
        attn_mode=attn_mode,
    )
    if pt_ckpt:
        load_pt_checkpoint(block, pt_ckpt)

    if dtype != ms.float32:
        amp.auto_mixed_precision(block, amp_level="O2", dtype=dtype)

    out = block(x, t, mask)
    print(out.shape)
    print(out.mean(), out.std())

    if pt_np:
        pt_out = np.load(pt_np)
        diff = _diff_res(out.asnumpy(), pt_out)
        print(diff)


def test_hyvtransformer(pt_ckpt=None, pt_np=None, debug=True, dtype=ms.float32, depth=None):
    # args
    args = edict()
    DEBUG_CONFIG = {
        "HYVideo-T/2": {
            "mm_double_blocks_depth": 1,
            "mm_single_blocks_depth": 1,
            "rope_dim_list": [4, 14, 14],  # [16, 56, 56], list sum = head_dim = pe_dim
            "hidden_size": 6 * 32,
            "heads_num": 6,
            "mlp_width_ratio": 1,
        },
    }
    enable_ms_amp = True
    model_cfg = DEBUG_CONFIG if debug else HUNYUAN_VIDEO_CONFIG
    args.model = "HYVideo-T/2"
    if depth is not None:
        model_cfg[args.model]["mm_double_blocks_depth"] = depth
        model_cfg[args.model]["mm_single_blocks_depth"] = depth

    if debug:
        args.text_states_dim = 64
        args.text_states_dim_2 = 24
        in_channels = 4
    else:
        args.text_states_dim = 4096
        args.text_states_dim_2 = 768
        in_channels = 16

    # shape
    token_shape = (bs, max_text_len, llm_emb_dim) = 1, 32, args.text_states_dim
    latent_shape = (bs, C, T, H, W) = (bs, in_channels, 5, 8, 8)
    clip_txt_len, clip_emb_dim = 18, args.text_states_dim_2
    patch_size = 2
    S_vid = T * (H // patch_size) * (W // patch_size)
    num_heads = model_cfg[args.model]["heads_num"]
    hidden_size = model_cfg[args.model]["hidden_size"]
    pe_dim = head_dim = hidden_size // num_heads
    freqs_cos_shape = (S_vid, head_dim)

    # np
    video_latent = np.random.normal(size=latent_shape).astype(np.float32)
    t = np.array([1000.0 for _ in range(bs)], dtype=np.float32)
    text_states = np.random.normal(size=token_shape).astype(np.float32)
    text_mask = np.zeros(shape=(bs, max_text_len), dtype=np.int32)  #
    text_states_2 = np.random.normal(size=(bs, clip_emb_dim)).astype(np.float32)  # [1, 768]
    freqs_cos = np.random.normal(size=freqs_cos_shape).astype(np.float32)
    freqs_sin = np.random.normal(size=freqs_cos_shape).astype(np.float32)
    guidance = np.array([7.0 * 1000 for _ in range(bs)], dtype=np.float32)
    text_mask[0, :4] = 1

    # i2v
    # semantic_images = np.random.normal(size=(bs, 3, T * 4, H * 8, W * 8)).astype(np.float32)
    # img_latents = np.random.normal(size=(bs, C, 1, H, W)).astype(np.float32)
    # i2v_mode = True
    # i2v_stability = True

    # tensor
    video_latent = ms.Tensor(video_latent)
    t = ms.Tensor(t)
    text_states = ms.Tensor(text_states)
    text_mask = ms.Tensor(text_mask)
    text_states_2 = ms.Tensor(text_states_2)
    freqs_cos = ms.Tensor(freqs_cos)
    freqs_sin = ms.Tensor(freqs_sin)
    guidance = ms.Tensor(guidance)

    # model
    factor_kwargs = {"dtype": dtype, "i2v_condition_type": "token_replace", "guidance_embed": True}
    net = HYVideoDiffusionTransformer(
        text_states_dim=args.text_states_dim,
        text_states_dim_2=args.text_states_dim_2,
        in_channels=C,
        use_conv2d_patchify=True,
        attn_mode="flash",
        **model_cfg[args.model],
        **factor_kwargs,
    )
    if dtype != ms.float32:
        set_model_param_dtype(net, dtype=dtype)

    if pt_ckpt:
        net.load_from_checkpoint(pt_ckpt)

    if enable_ms_amp and dtype != ms.float32:
        from hyvideo.modules.embed_layers import SinusoidalEmbedding
        from hyvideo.modules.norm_layers import LayerNorm, RMSNorm

        from mindone.utils.amp import auto_mixed_precision

        whitelist_ops = [
            LayerNorm,
            RMSNorm,
            SinusoidalEmbedding,
        ]
        print("custom fp32 cell for dit: ", whitelist_ops)
        net = auto_mixed_precision(net, amp_level="O2", dtype=dtype, custom_fp32_cells=whitelist_ops)

    # run

    if pt_np:
        if pt_np.endswith(".npy"):
            pt_out = np.load(pt_np)
            diff = _diff_res(out.asnumpy(), pt_out)
            print(diff)
        elif pt_np.endswith(".npz"):
            data = np.load(pt_np)

            latent_model_input = ms.Tensor(data["latent_model_input"], dtype=dtype)
            t_expand = ms.Tensor(data["t_expand"], dtype=ms.int32)
            prompt_embeds = ms.Tensor(data["prompt_embeds"], dtype=dtype)
            prompt_mask = ms.Tensor(data["prompt_mask"], dtype=ms.bool_)
            prompt_embeds_2 = ms.Tensor(data["prompt_embeds_2"], dtype=dtype)
            freqs_cos = ms.Tensor(data["freqs_cos"], dtype=ms.float32)
            freqs_sin = ms.Tensor(data["freqs_sin"], dtype=ms.float32)
            guidance_expand = ms.Tensor(data["guidance_expand"], dtype=ms.float32)

            noise_pred_ms = (
                net(
                    latent_model_input,
                    t_expand,
                    text_states=prompt_embeds,
                    text_mask=prompt_mask,
                    text_states_2=prompt_embeds_2,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    guidance=guidance_expand,
                )
                .float()
                .asnumpy()
            )
            noise_pred_torch = data["noise_pred"]
            diff = _diff_res(noise_pred_ms, noise_pred_torch)
            print(diff)
    else:
        start = time.time()
        out = net(video_latent, t, text_states, text_mask, text_states_2, freqs_cos, freqs_sin, guidance)
        print("time cost: ", time.time() - start)

        print(out.shape)
        print(out.mean(), out.std())


def test_nd_rope():
    latents_size = [16, 32, 32]
    patch_size = [1, 2, 2]
    rope_sizes = [s // patch_size[idx] for idx, s in enumerate(latents_size)]
    target_ndim = 3
    rope_dim_list = [16, 56, 56]
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=256,
        use_real=True,
        theta_rescale_factor=1,
    )

    print(freqs_cos.sum(), freqs_cos.std())
    print(freqs_sin.sum(), freqs_sin.std())


if __name__ == "__main__":
    ms.set_context(mode=1)
    # ms.set_context(mode=0, jit_syntax_level=ms.STRICT)
    # test_attn()
    test_dualstream_block("ckpts/transformer_depth1.pt", "forward_double_block_torch.npz")
    # test_singlestream_block('tests/single_stream.pth', 'tests/pt_single_stream.npy')
    # test_token_refiner('tests/token_refiner.pth', 'tests/pt_token_refiner.npy')
    # test_token_refiner('tests/token_refiner.pth', 'tests/pt_token_refiner.npy', attn_mode='vanilla', dtype=ms.float16)
    # test_hyvtransformer('tests/dit_tiny.pt', 'tests/pt_hyvtransformer.npy')

    # test_hyvtransformer()
    test_hyvtransformer(
        pt_ckpt="ckpts/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt", dtype=ms.bfloat16, debug=False
    )
    # test_hyvtransformer(pt_ckpt='ckpts/transformer_depth1.pt', pt_np='tests/pt_pretrained_hyvtransformer_ge.npy', dtype=ms.float32, debug=False, depth=1)

    # test_nd_rope()
