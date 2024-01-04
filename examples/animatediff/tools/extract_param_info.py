import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))


def extract_torch_mm():
    import torch

    pt_mm_path = "/home/AnimateDiff/models/Motion_Module/animatediff/mm_sd_v15_v2.ckpt"
    pt_mm_sd = torch.load(pt_mm_path)

    print(
        pt_mm_sd[
            "down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.to_q.weight"
        ].sum()
    )

    num_attn_layers = 0
    tot_params = 0
    for pname in pt_mm_sd:
        print(f"{pname}#{tuple(pt_mm_sd[pname].size())}#{pt_mm_sd[pname].dtype}")
        tot_params += 1
        if "to_q.weight" in pname:
            num_attn_layers += 1


def extract_ms_sd_mm():
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf

    model_config_path = "configs/stable_diffusion/v1-inference-unet3d.yaml"
    sd_config = OmegaConf.load(model_config_path)
    model = instantiate_from_config(sd_config.model)

    cnt = 0
    for param in model.get_parameters():
        if "temporal_transformer." in param.name:
            print(f"{param.name}#{param.shape}#{param.dtype}")
            cnt += 1

    print("Num temporla param: ", cnt)
    assert cnt // 28 == 21, "expect 588 params for 21 mm"


def extract_ms_unet_mm():
    from ldm.modules.diffusionmodules.unet3d import UNet3DModel

    unet3d = UNet3DModel(
        image_size=64,
        in_channels=4,
        model_channels=320,
        out_channels=320,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        dropout=0.0,
        channel_mult=(1, 2, 4, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=8,
        num_head_channels=-1,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=768,
        n_embed=None,
        legacy=False,
        use_linear_in_transformer=False,
        cross_frame_attention=False,
        unet_chunk_size=2,
        adm_in_channels=None,
        use_inflated_groupnorm=True,
        use_motion_module=True,
        motion_module_resolutions=(1, 2, 4, 8),
        motion_module_mid_block=True,
        motion_module_decoder_only=False,
        motion_module_type="Vanilla",
        motion_module_kwargs={
            "num_attention_heads": 8,
            "num_transformer_block": 1,
            "attention_block_types": ["Temporal_Self", "Temporal_Self"],
            "temporal_position_encoding": True,
            "temporal_position_encoding_max_len": 32,
            "temporal_attention_dim_div": 1,
        },
        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False,
    )

    cnt = 0
    for param in unet3d.get_parameters():
        if "temporal_transformer." in param.name:
            print(f"{param.name}#{param.shape}#{param.dtype}")
            cnt += 1

    print(cnt)


if __name__ == "__main__":
    extract_ms_sd_mm()
