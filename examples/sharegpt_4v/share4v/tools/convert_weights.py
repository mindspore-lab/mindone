import argparse
import os

import torch

import mindspore as ms


def load_torch_ckpt(ckpt_file):
    source_data = torch.load(ckpt_file, map_location="cpu")
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    return source_data


def convert_pt_name_to_ms(content: str) -> str:
    # ShareGPT4V
    if content.startswith("model.layers"):
        content = content.replace("self_attn.q_proj", "attention.wq")
        content = content.replace("self_attn.k_proj", "attention.wk")
        content = content.replace("self_attn.v_proj", "attention.wv")
        content = content.replace("self_attn.o_proj", "attention.wo")
        content = content.replace("mlp.gate_proj", "feed_forward.w1")
        content = content.replace("mlp.up_proj", "feed_forward.w3")
        content = content.replace("mlp.down_proj", "feed_forward.w2")
        content = content.replace("input_layernorm", "attention_norm")
        content = content.replace("post_attention_layernorm", "ffn_norm")
        # self_attn.rotary_emb.inv_freq doesn't exist in mindspore

    elif content.startswith("vision_model"):
        content = content.replace("vision_model", "vision_tower.vision_model")
        content = content.replace("position_embedding.weight", "position_embedding.embedding_table")
        if "norm" in content:
            # layer_norm
            content = content.replace("weight", "gamma")
            content = content.replace("bias", "beta")
    else:
        content = content.replace("model.norm", "model.norm_out")

    # this part used for self-defined llama
    content = content.replace("embed_tokens.weight", "embed_tokens.embedding_table")

    return content


def torch_to_ms_weight(source_fp_ls, target_fp):
    target_data = []
    for source_fp in source_fp_ls:
        source_data = load_torch_ckpt(source_fp)
        for _name_pt in source_data:
            _name_ms = convert_pt_name_to_ms(_name_pt)
            _source_data = ms.Tensor(source_data[_name_pt].float().cpu().detach().numpy())
            # if '.wq' in _name_ms or '.wk' in _name_ms:
            #     _source_data = permute(_source_data)
            target_data.append({"name": _name_ms, "data": _source_data})
    ms.save_checkpoint(target_data, target_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def list_of_address(x):
        return x.split(",")

    parser.add_argument("--source", type=list_of_address, help="path to source torch checkpoint")
    parser.add_argument(
        "--target",
        type=str,
        help="Filename to save. Specify folder, e.g., ./models, or file path which ends with .ckpt, e.g., ./models/ms_model.ckpt",
    )
    args = parser.parse_args()
    print(args.source)
    print(args.target)
    for i in args.source:
        if not os.path.exists(i):
            raise ValueError(f"The provided source file {i} does not exist!")

    if not args.target.endswith(".ckpt"):
        os.makedirs(args.target, exist_ok=True)
        target_fp = os.path.join(args.target, "mindspore_llama_model.ckpt")
    else:
        target_fp = args.target

    torch_to_ms_weight(args.source, target_fp)
