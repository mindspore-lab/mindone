import mindspore as ms


def init_and_save_weight(ckpt_path: str) -> None:
    """
    ckpt_path: path to SDXL pretrained ckpt (sd_xl_base_1.0_ms.ckpt)
    """
    data = ms.load_checkpoint(ckpt_path)
    records = list()
    for k, v in data.items():
        records.append({"name": k, "data": v})
        # duplicated
        if ".output_blocks." in k or ".out." in k:
            continue
        records.append({"name": k.replace("diffusion_model.", "diffusion_model.controlnet."), "data": v})

    ms.save_checkpoint(records, ckpt_path.replace(".ckpt", "_controlnet_init.ckpt"))


if __name__ == "__main__":
    init_and_save_weight("checkpoints/sd_xl_base_1.0_ms.ckpt")
