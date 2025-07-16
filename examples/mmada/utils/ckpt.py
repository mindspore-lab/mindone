import mindspore as ms


def init_from_ckpt(model, path, ignore_keys=list()):
    sd = ms.load_checkpoint(path)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, sd)
    assert (
        len(param_not_load) == len(ckpt_not_load) == 0
    ), "Exist ckpt params not loaded: {} (total: {})\nor net params not loaded: {} (total: {})".format(
        ckpt_not_load, len(ckpt_not_load), param_not_load, len(param_not_load)
    )
