from hyvideo.constants import PRECISION_TO_TYPE

import mindspore as ms
from mindspore.communication.management import GlobalComm

from mindone.trainers.zero import prepare_network

from ..utils.helpers import set_model_param_dtype
from .models import HUNYUAN_VIDEO_CONFIG, HYVideoDiffusionTransformer


def load_model(
    name,
    in_channels,
    out_channels,
    factor_kwargs,
    zero_stage=None,
    text_states_dim: int = 4096,
    text_states_dim_2: int = 768,
):
    """load hunyuan video model

    Args:
        in_channels (int): input channels number
        out_channels (int): output channels number
        zero_stage (int, optional): zero stage. Defaults to None.
        text_states_dim (int, optional): text states dim of text encoder 1. Defaults to 4096.
        text_states_dim (int ,optional): text states dim of text encoder 2. Defaults to 768.
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    if name in HUNYUAN_VIDEO_CONFIG.keys():
        model = HYVideoDiffusionTransformer(
            text_states_dim=text_states_dim,
            text_states_dim_2=text_states_dim_2,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[name],
            **factor_kwargs,
        )
        if zero_stage is not None:
            assert zero_stage in [0, 1, 2, 3], "zero_stage should be in [0, 1, 2, 3]"
            model = prepare_network(
                model,
                zero_stage=zero_stage,
                op_group=GlobalComm.WORLD_COMM_GROUP,
            )

        # half model parameter
        dtype = factor_kwargs["dtype"]
        if isinstance(dtype, str):
            dtype = PRECISION_TO_TYPE[dtype]
        if dtype != ms.float32:
            set_model_param_dtype(model, dtype=dtype)

        return model
    else:
        raise NotImplementedError()
