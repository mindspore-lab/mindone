import argparse

from safetensors.torch import load_file as load_safetensors

GLOBAL_MAPPING = {
    # applied to all param names
    "beta": "bias",
    "gamma": "weight",
}

PARAMS_MAPPING = {
    "model.diffusion_model.label_emb.0.0": "add_embedding.linear_1",  # keys found in mindone sdxl weight, not found in mindone control+sdxl weight
    "model.diffusion_model.label_emb.0.2": "add_embedding.linear_2",  # keys found in mindone sdxl weight, not found in mindone control+sdxl weight
    "model.diffusion_model.controlnet.input_hint_block.0": "controlnet_cond_embedding.conv_in",
    "model.diffusion_model.controlnet.input_hint_block.2": "controlnet_cond_embedding.blocks.0",
    "model.diffusion_model.controlnet.input_hint_block.4": "controlnet_cond_embedding.blocks.1",
    "model.diffusion_model.controlnet.input_hint_block.6": "controlnet_cond_embedding.blocks.2",
    "model.diffusion_model.controlnet.input_hint_block.8": "controlnet_cond_embedding.blocks.3",
    "model.diffusion_model.controlnet.input_hint_block.10": "controlnet_cond_embedding.blocks.4",
    "model.diffusion_model.controlnet.input_hint_block.12": "controlnet_cond_embedding.blocks.5",
    "model.diffusion_model.controlnet.input_hint_block.14": "controlnet_cond_embedding.conv_out",
    "model.diffusion_model.controlnet.zero_convs.0.0": "controlnet_down_blocks.0",
    "model.diffusion_model.controlnet.zero_convs.1.0": "controlnet_down_blocks.1",
    "model.diffusion_model.controlnet.zero_convs.2.0": "controlnet_down_blocks.2",
    "model.diffusion_model.controlnet.zero_convs.3.0": "controlnet_down_blocks.3",
    "model.diffusion_model.controlnet.zero_convs.4.0": "controlnet_down_blocks.4",
    "model.diffusion_model.controlnet.zero_convs.5.0": "controlnet_down_blocks.5",
    "model.diffusion_model.controlnet.zero_convs.6.0": "controlnet_down_blocks.6",
    "model.diffusion_model.controlnet.zero_convs.7.0": "controlnet_down_blocks.7",
    "model.diffusion_model.controlnet.zero_convs.8.0": "controlnet_down_blocks.8",
    "model.diffusion_model.controlnet.middle_block_out.0": "controlnet_mid_block",
    "model.diffusion_model.controlnet.input_blocks.0.0": "conv_in",
    "model.diffusion_model.controlnet.input_blocks.3.0.op": "down_blocks.0.downsamplers.0.conv",
    "model.diffusion_model.controlnet.input_blocks.1.0.in_layers.2": "down_blocks.0.resnets.0.conv1",
    "model.diffusion_model.controlnet.input_blocks.1.0.out_layers.3": "down_blocks.0.resnets.0.conv2",
    "model.diffusion_model.controlnet.input_blocks.1.0.in_layers.0": "down_blocks.0.resnets.0.norm1",
    "model.diffusion_model.controlnet.input_blocks.1.0.out_layers.0": "down_blocks.0.resnets.0.norm2",
    "model.diffusion_model.controlnet.input_blocks.1.0.emb_layers.1": "down_blocks.0.resnets.0.time_emb_proj",
    "model.diffusion_model.controlnet.input_blocks.2.0.in_layers.2": "down_blocks.0.resnets.1.conv1",
    "model.diffusion_model.controlnet.input_blocks.2.0.out_layers.3": "down_blocks.0.resnets.1.conv2",
    "model.diffusion_model.controlnet.input_blocks.2.0.in_layers.0": "down_blocks.0.resnets.1.norm1",
    "model.diffusion_model.controlnet.input_blocks.2.0.out_layers.0": "down_blocks.0.resnets.1.norm2",
    "model.diffusion_model.controlnet.input_blocks.2.0.emb_layers.1": "down_blocks.0.resnets.1.time_emb_proj",
    "model.diffusion_model.controlnet.input_blocks.4.1": "down_blocks.1.attentions.0",  # 46 transformer_blocks
    "model.diffusion_model.controlnet.input_blocks.5.1": "down_blocks.1.attentions.1",  # 46 transformer_blocks
    "model.diffusion_model.controlnet.input_blocks.6.0.op": "down_blocks.1.downsamplers.0.conv",
    "model.diffusion_model.controlnet.input_blocks.4.0.in_layers.2": "down_blocks.1.resnets.0.conv1",
    "model.diffusion_model.controlnet.input_blocks.4.0.out_layers.3": "down_blocks.1.resnets.0.conv2",
    "model.diffusion_model.controlnet.input_blocks.4.0.skip_connection": "down_blocks.1.resnets.0.conv_shortcut",
    "model.diffusion_model.controlnet.input_blocks.4.0.in_layers.0": "down_blocks.1.resnets.0.norm1",
    "model.diffusion_model.controlnet.input_blocks.4.0.out_layers.0": "down_blocks.1.resnets.0.norm2",
    "model.diffusion_model.controlnet.input_blocks.4.0.emb_layers.1": "down_blocks.1.resnets.0.time_emb_proj",
    "model.diffusion_model.controlnet.input_blocks.5.0.in_layers.2": "down_blocks.1.resnets.1.conv1",
    "model.diffusion_model.controlnet.input_blocks.5.0.out_layers.3": "down_blocks.1.resnets.1.conv2",
    "model.diffusion_model.controlnet.input_blocks.5.0.in_layers.0": "down_blocks.1.resnets.1.norm1",
    "model.diffusion_model.controlnet.input_blocks.5.0.out_layers.0": "down_blocks.1.resnets.1.norm2",
    "model.diffusion_model.controlnet.input_blocks.5.0.emb_layers.1": "down_blocks.1.resnets.1.time_emb_proj",
    "model.diffusion_model.controlnet.input_blocks.7.1": "down_blocks.2.attentions.0",  # 206 transformer_blocks
    "model.diffusion_model.controlnet.input_blocks.8.1": "down_blocks.2.attentions.1",  # 206 transformer_blocks
    "model.diffusion_model.controlnet.input_blocks.7.0.in_layers.2": "down_blocks.2.resnets.0.conv1",
    "model.diffusion_model.controlnet.input_blocks.7.0.out_layers.3": "down_blocks.2.resnets.0.conv2",
    "model.diffusion_model.controlnet.input_blocks.7.0.skip_connection": "down_blocks.2.resnets.0.conv_shortcut",
    "model.diffusion_model.controlnet.input_blocks.7.0.in_layers.0": "down_blocks.2.resnets.0.norm1",
    "model.diffusion_model.controlnet.input_blocks.7.0.out_layers.0": "down_blocks.2.resnets.0.norm2",
    "model.diffusion_model.controlnet.input_blocks.7.0.emb_layers.1": "down_blocks.2.resnets.0.time_emb_proj",
    "model.diffusion_model.controlnet.input_blocks.8.0.in_layers.2": "down_blocks.2.resnets.1.conv1",
    "model.diffusion_model.controlnet.input_blocks.8.0.out_layers.3": "down_blocks.2.resnets.1.conv2",
    "model.diffusion_model.controlnet.input_blocks.8.0.in_layers.0": "down_blocks.2.resnets.1.norm1",
    "model.diffusion_model.controlnet.input_blocks.8.0.out_layers.0": "down_blocks.2.resnets.1.norm2",
    "model.diffusion_model.controlnet.input_blocks.8.0.emb_layers.1": "down_blocks.2.resnets.1.time_emb_proj",
    "model.diffusion_model.controlnet.middle_block.1": "mid_block.attentions.0",  # 206 transformer_blocks
    "model.diffusion_model.controlnet.middle_block.0.in_layers.2": "mid_block.resnets.0.conv1",
    "model.diffusion_model.controlnet.middle_block.0.out_layers.3": "mid_block.resnets.0.conv2",
    "model.diffusion_model.controlnet.middle_block.0.in_layers.0": "mid_block.resnets.0.norm1",
    "model.diffusion_model.controlnet.middle_block.0.out_layers.0": "mid_block.resnets.0.norm2",
    "model.diffusion_model.controlnet.middle_block.0.emb_layers.1": "mid_block.resnets.0.time_emb_proj",
    "model.diffusion_model.controlnet.middle_block.2.in_layers.2": "mid_block.resnets.1.conv1",
    "model.diffusion_model.controlnet.middle_block.2.out_layers.3": "mid_block.resnets.1.conv2",
    "model.diffusion_model.controlnet.middle_block.2.in_layers.0": "mid_block.resnets.1.norm1",
    "model.diffusion_model.controlnet.middle_block.2.out_layers.0": "mid_block.resnets.1.norm2",
    "model.diffusion_model.controlnet.middle_block.2.emb_layers.1": "mid_block.resnets.1.time_emb_proj",
    "model.diffusion_model.controlnet.time_embed.0": "time_embedding.linear_1",
    "model.diffusion_model.controlnet.time_embed.2": "time_embedding.linear_2",
}


def replace(content, old, new):
    content = content.replace("beta", "bias")
    content = content.replace("gamma", "weight")
    content = content.replace(old, new)
    return content


def dump_key_mapping(args):
    with open(args.ms_param_sdxl_controlnet) as f:
        key_ms_sdxl_controlnet = f.readlines()

    mapping = dict()
    for line in key_ms_sdxl_controlnet:
        key_ms = line.split(":")[0]
        for str_ms, str_torch in PARAMS_MAPPING.items():
            if key_ms.startswith(str_ms):
                mapping[key_ms] = replace(key_ms, str_ms, str_torch)

    with open(args.mapping_filepath, "w") as f:
        for k, v in mapping.items():
            f.write(k + ":" + v + "\n")
    print(f"INFO: Key mapping file {args.mapping_filepath} is dumped")


def weight2yaml(args):
    weight = load_safetensors(args.weight_torch_controlnet)
    with open(args.torch_param_sdxl_controlnet, "w") as f:
        for k, v in weight.items():
            f.write(k + ":" + str(tuple(v.shape)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dump controlnet weight param mapping between diffusers and mindone")
    parser.add_argument(
        "--mapping_filepath",
        type=str,
        default="./controlnet_ms2torch_mapping.yaml",
        help="path to param name mapping file for controlnet weight conversion",
    )
    parser.add_argument(
        "--ms_param_sdxl_controlnet",
        type=str,
        default="./ms_param_sdxl_base_controlnet.yaml",
        help="path to file of param name and shape of mindone controlnet + sdxl",
    )
    parser.add_argument(
        "--torch_param_sdxl_controlnet",
        type=str,
        default="./torch_param_sdxl_base_controlnet.yaml",
        help="path to file of param name and shape of diffusers controlnet + sdxl, "
        "diffusers only releases the weight of controlnet part",
    )
    parser.add_argument(
        "--weight_torch_controlnet",
        type=str,
        required=True,
        help="path to controlnet weight from torch (diffusion_pytorch_model.safetensors)",
    )

    args, _ = parser.parse_known_args()
    weight2yaml(args)
    dump_key_mapping(args)
