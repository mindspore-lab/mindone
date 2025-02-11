import logging
import os
from typing import Dict, List, Union

import numpy as np
from library import model_util, sdxl_original_unet
from safetensors import numpy
from transformers import CLIPTextConfig

import mindspore as ms
from mindspore import Parameter, nn, ops

from mindone.diffusers import AutoencoderKL
from mindone.transformers import CLIPTextModel, CLIPTextModelWithProjection

logger = logging.getLogger(__name__)

VAE_SCALE_FACTOR = 0.13025
MODEL_VERSION_SDXL_BASE_V1_0 = "sdxl_base_v1-0"

# # Diffusersの設定を読み込むための参照モデル
# DIFFUSERS_REF_MODEL_ID_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"


def load_file(filename: Union[str, os.PathLike]) -> Dict[str, ms.Tensor]:
    """
    Loads a safetensors file into mindspore format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, ms.Tensor]`: dictionary that contains name as key, value as `ms.Tensor`

    Example:

    ```python
    from safetensors_ms import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    flat = numpy.load_file(filename)
    output = _np2ms(flat)
    return output


def _np2ms(np_dict: Dict[str, np.ndarray]) -> Dict[str, ms.Tensor]:
    for k, v in np_dict.items():
        np_dict[k] = ms.Parameter(v, name=k)
    return np_dict


def convert_sdxl_text_encoder_2_checkpoint(checkpoint, max_length):
    SDXL_KEY_PREFIX = "conditioner.embedders.1.model."

    # SD2のと、基本的には同じ。logit_scaleを後で使うので、それを追加で返す
    # logit_scaleはcheckpointの保存時に使用する
    def convert_key(key):
        # common conversion
        key = key.replace(SDXL_KEY_PREFIX + "transformer.", "text_model.encoder.")
        key = key.replace(SDXL_KEY_PREFIX, "text_model.")

        if "resblocks" in key:
            # resblocks conversion
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.")
                key = key.replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in SD: {key}")
        elif ".positional_embedding" in key:
            key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif ".text_projection" in key:
            key = key.replace("text_model.text_projection", "text_projection.weight")
        elif ".logit_scale" in key:
            key = None  # 後で処理する
        elif ".token_embedding" in key:
            key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif ".ln_final" in key:
            key = key.replace(".ln_final", ".final_layer_norm")
        # ckpt from comfy has this key: text_model.encoder.text_model.embeddings.position_ids
        elif ".embeddings.position_ids" in key:
            key = None  # remove this key: position_ids is not used in newer transformers
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if ".resblocks" in key and ".attn.in_proj_" in key:
            # 三つに分割
            values = ops.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace(SDXL_KEY_PREFIX + "transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = Parameter(values[0])
            new_sd[key_pfx + "k_proj" + key_suffix] = Parameter(values[1])
            new_sd[key_pfx + "v_proj" + key_suffix] = Parameter(values[2])

    # logit_scale はDiffusersには含まれないが、保存時に戻したいので別途返す
    logit_scale = checkpoint.get(SDXL_KEY_PREFIX + "logit_scale", None)

    # temporary workaround for text_projection.weight.weight for Playground-v2
    if "text_projection.weight.weight" in new_sd:
        logger.info(
            "convert_sdxl_text_encoder_2_checkpoint: convert text_projection.weight.weight to text_projection.weight"
        )
        new_sd["text_projection.weight"] = new_sd["text_projection.weight.weight"]
        del new_sd["text_projection.weight.weight"]

    return new_sd, logit_scale


# load state_dict without allocating new tensors
def _load_state_dict_on_device(model, state_dict, dtype=None):
    # dtype will use fp32 as default
    missing_keys = list(model.state_dict().keys() - state_dict.keys())
    unexpected_keys = list(state_dict.keys() - model.state_dict().keys())

    # similar to model.load_state_dict()
    if not missing_keys and not unexpected_keys:
        # for k in list(state_dict.keys()):
        #     set_module_tensor_to_device(model, k, value=state_dict.pop(k), dtype=dtype)
        return "<All keys matched successfully>"

    # error_msgs
    error_msgs: List[str] = []
    if missing_keys:
        error_msgs.insert(
            0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys))
        )
    if unexpected_keys:
        error_msgs.insert(
            0, "Unexpected key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in unexpected_keys))
        )

    raise RuntimeError(
        "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
    )


def load_models_from_sdxl_checkpoint(ckpt_path, dtype=None):
    # model_version is reserved for future use
    # dtype is used for full_fp16/bf16 integration. Text Encoder will remain fp32, because it runs on CPU when caching

    # Load the state dict
    if ckpt_path.endswith(".ckpt"):
        state_dict = ms.load_checkpoint(ckpt_path)
    else:
        state_dict = load_file(ckpt_path)  # prevent device invalid Error
        logger.info("load from safetensor")

    epoch = None
    global_step = None

    # U-Net
    logger.info("building U-Net")
    unet = sdxl_original_unet.SdxlUNet2DConditionModel()

    logger.info("loading U-Net from checkpoint")
    unet_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = state_dict.pop(k)
    if "safetensors" in ckpt_path:
        unet_sd = _convert_state_dict(unet, unet_sd)
    local_states = {k: v for k, v in unet.parameters_and_names()}
    for k, v in unet_sd.items():
        for k in local_states:
            v.set_dtype(local_states[k].dtype)
        else:
            pass
    info, _ = ms.load_param_into_net(unet, unet_sd)
    logger.info(f"U-Net: {info}")

    # Text Encoders
    logger.info("building text encoders")

    # Text Encoder 1 is same to Stability AI's SDXL
    text_model1_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=768,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    text_model1 = CLIPTextModel(text_model1_cfg)

    # Text Encoder 2 is different from Stability AI's SDXL. SDXL uses open clip, but we use the model from HuggingFace.
    # Note: Tokenizer from HuggingFace is different from SDXL. We must use open clip's tokenizer.
    text_model2_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=1280,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    text_model2 = CLIPTextModelWithProjection(text_model2_cfg)

    logger.info("loading text encoders from checkpoint")
    te1_sd = {}
    te2_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("conditioner.embedders.0.transformer."):
            te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = state_dict.pop(k)
        elif k.startswith("conditioner.embedders.1.model."):
            te2_sd[k] = state_dict.pop(k)

    # 最新の transformers では position_ids を含むとエラーになるので削除 / remove position_ids for latest transformers
    if "text_model.embeddings.position_ids" in te1_sd:
        te1_sd.pop("text_model.embeddings.position_ids")
    if "safetensors" in ckpt_path:
        tel_sd = _convert_state_dict(text_model1, te1_sd)
    local_states = {k: v for k, v in text_model1.parameters_and_names()}
    for k, v in tel_sd.items():
        for k in local_states:
            v.set_dtype(local_states[k].dtype)
        else:
            pass
    info1, _ = ms.load_param_into_net(text_model1, te1_sd)
    logger.info(f"text encoder 1: {info1}")

    converted_sd, logit_scale = convert_sdxl_text_encoder_2_checkpoint(te2_sd, max_length=77)
    if "safetensors" in ckpt_path:
        converted_sd = _convert_state_dict(text_model2, converted_sd)
    local_states = {k: v for k, v in text_model2.parameters_and_names()}
    for k, v in converted_sd.items():
        for k in local_states:
            v.set_dtype(local_states[k].dtype)
        else:
            pass
    info2, _ = ms.load_param_into_net(text_model2, converted_sd)
    logger.info(f"text encoder 2: {info2}")

    # prepare vae
    logger.info("building VAE")
    vae_config = model_util.create_vae_diffusers_config()
    vae = AutoencoderKL(**vae_config)

    logger.info("loading VAE from checkpoint")
    converted_vae_checkpoint = model_util.convert_ldm_vae_checkpoint(state_dict, vae_config)
    info, _ = ms.load_param_into_net(vae, converted_vae_checkpoint)
    logger.info(f"VAE: {info}")

    ckpt_info = (epoch, global_step) if epoch is not None else None
    return text_model1, text_model2, vae, unet, logit_scale, ckpt_info


def make_unet_conversion_map():
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    return unet_conversion_map


def _get_pt2ms_mappings(m):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: ms.Parameter(
                ops.expand_dims(x, axis=-2), name=x.name
            )
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(cell, (nn.BatchNorm2d,)):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
    return mappings


def _convert_state_dict(m, state_dict_pt):
    if not state_dict_pt:
        return state_dict_pt
    pt2ms_mappings = _get_pt2ms_mappings(m)
    state_dict_ms = {}
    while state_dict_pt:
        name_pt, data_pt = state_dict_pt.popitem()
        name_ms, data_mapping = pt2ms_mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(data_pt)
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms


def convert_diffusers_unet_state_dict_to_sdxl(du_sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_map = {hf: sd for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(du_sd, conversion_map)


def convert_unet_state_dict(src_sd, conversion_map):
    converted_sd = {}
    for param in src_sd:
        src_key = param.name
        value = param.data
        # さすがに全部回すのは時間がかかるので右から要素を削りつつprefixを探す
        src_key_fragments = src_key.split(".")[:-1]  # remove weight/bias
        while len(src_key_fragments) > 0:
            src_key_prefix = ".".join(src_key_fragments) + "."
            if src_key_prefix in conversion_map:
                converted_prefix = conversion_map[src_key_prefix]
                converted_key = converted_prefix + src_key[len(src_key_prefix) :]
                converted_sd[converted_key] = value
                break
            src_key_fragments.pop(-1)
        assert len(src_key_fragments) > 0, f"key {src_key} not found in conversion map"

    return converted_sd
