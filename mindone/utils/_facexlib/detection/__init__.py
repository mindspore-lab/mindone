import mindspore

from ..utils import load_file_from_hf
from .retinaface import RetinaFace


def init_detection_model(model_name, half=False, model_rootpath=None, subfolder=None):
    repo_id = model_rootpath or "townwish/facexlib-weights"
    subfolder = subfolder or "detection"
    if model_name == "retinaface_resnet50":
        # model_url = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
        model = RetinaFace(network_name="resnet50", half=half)
        filename = "detection_Resnet50_Final.ckpt"
    elif model_name == "retinaface_mobile0.25":
        # model_url = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth"
        model = RetinaFace(network_name="mobile0.25", half=half)
        filename = "detection_mobilenet0.25_Final.ckpt"
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
