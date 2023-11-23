def save_params(model, out_path):
    param_list = []
    for param in model.get_parameters():
        line = param.name + "#" + \
            str(param.dtype) + "#" + str(tuple(param.shape))
        param_list.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(param_list))


def set_params(model, pt_weights_path, save_path=None):
    import pickle
    import mindspore as ms

    with open(pt_weights_path, 'rb') as f:
        pt_weights = pickle.load(f)

    for i, ms_param in enumerate(model.get_parameters()):
        if len(ms_param.shape) == len(pt_weights[i].shape):
            ms_param.set_data(ms.Tensor(pt_weights[i], dtype=ms.float32))
        else:
            ms_weight = ms.Tensor(pt_weights[i], dtype=ms.float32).unsqueeze(2)
            ms_param.set_data(ms_weight)

    if save_path:
        ms.save_checkpoint(model, save_path)
        print(f"checkpoint is saved to {save_path}.")

    return model
