# from .imagebase import imagebase_ae, imagebase_ae_stride, imagebase_ae_channel
from .videobase import (  # VQVAEConfiguration,; VQVAEModel,; VQVAETrainer,; CausalVQVAEConfiguration,; CausalVQVAEModel,; CausalVQVAETrainer,
    videobase_ae,
    videobase_ae_channel,
    videobase_ae_stride,
    videobase_ae_yaml,
)

ae_stride_config = {}
# ae_stride_config.update(imagebase_ae_stride)
ae_stride_config.update(videobase_ae_stride)

ae_channel_config = {}
# ae_channel_config.update(imagebase_ae_channel)
ae_channel_config.update(videobase_ae_channel)

# def getae(args):
#     """deprecation"""
#     ae = imagebase_ae.get(args.ae, None) or videobase_ae.get(args.ae, None)
#     assert ae is not None
#     return ae(args.ae)


def getae_wrapper(ae):
    """deprecation"""
    # ae = imagebase_ae.get(ae, None) or videobase_ae.get(ae, None)
    ae = videobase_ae.get(ae, None)
    assert ae is not None
    return ae


def getae_model_config(ae):
    ae = videobase_ae_yaml.get(ae, None)
    return ae
