import torch

import mindspore as ms


def _load_torch_ckpt(ckpt_file):
    source_data = torch.load(ckpt_file, map_location="cpu")
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    return source_data


def _load_huggingface_safetensor(ckpt_file):
    from safetensors import safe_open

    db_state_dict = {}
    with safe_open(ckpt_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            db_state_dict[key] = f.get_tensor(key)
    return db_state_dict


LOAD_PYTORCH_FUNCS = {"others": _load_torch_ckpt, "safetensors": _load_huggingface_safetensor}


def load_torch_ckpt(ckpt_path):
    extension = ckpt_path.split(".")[-1]
    if extension not in LOAD_PYTORCH_FUNCS.keys():
        extension = "others"
    torch_params = LOAD_PYTORCH_FUNCS[extension](ckpt_path)
    return torch_params


def convert_pt_name_to_ms(content: str) -> str:
    # DiT embedding table name conversion
    # content = content.replace("y_embedder.embedding_table.weight", "y_embedder.embedding_table.embedding_table")
    content = content.replace("module.logit_scale", "logit_scale")
    content = content.replace("module.text", "text")
    content = content.replace("token_embedding.weight", "embedding_table")
    content = content.replace("text.transformer", "text.transformer_layer")
    content = content.replace("ln_1.weight", "ln_1.gamma")
    content = content.replace("ln_1.bias", "ln_1.beta")
    content = content.replace("ln_2.weight", "ln_2.gamma")
    content = content.replace("ln_2.bias", "ln_2.beta")
    content = content.replace(".mlp.", ".")
    content = content.replace("ln_final.weight", "ln_final.gamma")
    content = content.replace("ln_final.bias", "ln_final.beta")
    content = content.replace("module.visual", "visual.visual")
    content = content.replace("ln_pre.weight", "ln_pre.gamma")
    content = content.replace("ln_pre.bias", "ln_pre.beta")
    content = content.replace("ln_q.weight", "ln_q.gamma")
    content = content.replace("ln_q.bias", "ln_q.beta")
    content = content.replace("ln_k.weight", "ln_k.gamma")
    content = content.replace("ln_k.bias", "ln_k.beta")
    content = content.replace("ln_post.weight", "ln_post.gamma")
    content = content.replace("ln_post.bias", "ln_post.beta")
    content = content.replace("module.text_decoder", "text_decoder")
    content = content.replace("ln_1_kv.weight", "ln_1_kv.gamma")
    content = content.replace("ln_1_kv.bias", "ln_1_kv.beta")
    return content


def torch_to_ms_weight(source_fp, target_fp):
    source_data = load_torch_ckpt(source_fp)
    target_data = []
    for _name_pt in source_data:
        _name_ms = convert_pt_name_to_ms(_name_pt)
        _source_data = source_data[_name_pt].cpu().detach().numpy()
        target_data.append({"name": _name_ms, "data": ms.Tensor(_source_data)})
    ms.save_checkpoint(target_data, target_fp)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--source",
    #     "-s",
    #     type=str,
    #     help="path to source torch checkpoint, which ends with .pt",
    # )
    # parser.add_argument(
    #     "--target",
    #     "-t",
    #     type=str,
    #     help="Filename to save. Specify folder, e.g., ./models, or file path which ends with .ckpt, e.g., ./models/dit.ckpt",
    # )

    # args = parser.parse_args()

    # if not os.path.exists(args.source):
    #     raise ValueError(f"The provided source file {args.source} does not exist!")

    # if not args.target.endswith(".ckpt"):
    #     os.makedirs(args.target, exist_ok=True)
    #     target_fp = os.path.join(args.target, os.path.basename(args.source).split(".")[0] + ".ckpt")
    # else:
    #     target_fp = args.target

    # if os.path.exists(target_fp):
    #     print(f"Warnings: {target_fp} will be overwritten!")

    # torch_to_ms_weight(args.source, target_fp)
    # print(f"Converted weight saved to {target_fp}")
    source_fp = "/disk1/mindone/songyuanwei/models/open_clip_pytorch_model.bin"
    target = "/disk1/mindone/songyuanwei/models/coca_model.ckpt"
    torch_to_ms_weight(source_fp, target)
