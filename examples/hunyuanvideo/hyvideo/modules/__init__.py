import mindspore as ms
from mindspore.communication.management import GlobalComm

from mindone.trainers.zero import prepare_network

from ..utils.helpers import set_model_param_dtype
from .models import HUNYUAN_VIDEO_CONFIG, HYVideoDiffusionTransformer


def load_model(args, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    if args.model in HUNYUAN_VIDEO_CONFIG.keys():
        model = HYVideoDiffusionTransformer(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        if args.zero_stage is not None:
            assert args.zero_stage in [0, 1, 2, 3], "zero_stage should be in [0, 1, 2, 3]"
            model = prepare_network(
                model,
                zero_stage=args.zero_stage,
                op_group=GlobalComm.WORLD_COMM_GROUP,
            )

        # half model parameter
        dtype = factor_kwargs["dtype"]
        if dtype != ms.float32:
            set_model_param_dtype(model, dtype=dtype)

        return model
    else:
        raise NotImplementedError()
