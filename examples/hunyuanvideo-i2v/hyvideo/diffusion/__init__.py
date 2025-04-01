from .flow.transport import ModelType, PathType, SNRType, Transport, WeightType
from .pipelines import HunyuanVideoPipeline
from .schedulers import FlowMatchDiscreteScheduler


def create_transport(
    *,
    path_type,
    prediction,
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    snr_type="uniform",
    shift=1.0,
    video_shift=None,
    reverse=False,
):
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    if snr_type == "lognorm":
        snr_type = SNRType.LOGNORM
    elif snr_type == "uniform":
        snr_type = SNRType.UNIFORM
    else:
        raise ValueError(f"Invalid snr type {snr_type}")

    if video_shift is None:
        video_shift = shift

    path_choice = {
        "linear": PathType.LINEAR,
        "gvp": PathType.GVP,
        "vp": PathType.VP,
    }

    path_type = path_choice[path_type.lower()]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        snr_type=snr_type,
        shift=shift,
        video_shift=video_shift,
        reverse=reverse,
    )

    return state


def load_denoiser(args):
    if args.denoise_type == "flow":
        denoiser = create_transport(
            path_type=args.flow_path_type,
            prediction=args.flow_predict_type,
            loss_weight=args.flow_loss_weight,
            train_eps=args.flow_train_eps,
            sample_eps=args.flow_sample_eps,
            snr_type=args.flow_snr_type,
            shift=args.flow_shift,
            video_shift=args.flow_shift,
            reverse=args.flow_reverse,
        )
    else:
        raise ValueError(f"Unknown denoise type: {args.denoise_type}")
    return denoiser
