import mindspore

from ..utils import load_file_from_hf
from .bisenet import BiSeNet
from .parsenet import ParseNet


def init_parsing_model(model_name="bisenet", half=False, model_rootpath=None, subfolder=None):
    repo_id = model_rootpath or "townwish/facexlib-weights"
    subfolder = subfolder or "parsing"
    if model_name == "bisenet":
        # model_url = "https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth"
        model = BiSeNet(num_class=19)
        filename = "parsing_bisenet.ckpt"
    elif model_name == "parsenet":
        # model_url = "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        filename = "parsing_parsenet.ckpt"
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")

    model_path = load_file_from_hf(
        repo_id,
        subfolder=subfolder,
        filename=filename,
    )
    load_net = mindspore.load_checkpoint(model_path)
    mindspore.load_param_into_net(model, load_net)
    model.set_train(False)
    return model
