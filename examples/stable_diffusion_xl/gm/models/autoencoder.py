# reference to https://github.com/Stability-AI/generative-models
from typing import Any, Dict, Tuple, Union

from gm.modules.distributions.distributions import DiagonalGaussianDistribution
from gm.util import default, instantiate_from_config
from omegaconf import ListConfig

import mindspore as ms
from mindspore import Tensor, nn


class AbstractAutoencoder(nn.Cell):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        monitor: Union[None, str] = None,
        input_key: str = "jpg",
        ckpt_path: Union[None, str] = None,
        ignore_keys: Union[Tuple, list, ListConfig] = (),
    ):
        super().__init__()
        self.input_key = input_key
        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path, ignore_keys=ignore_keys)

    def load_checkpoint(self, path: str, ignore_keys: Union[Tuple, list, ListConfig] = tuple()) -> None:
        assert path.endswith(".ckpt"), f"checkpoint path expect '*.ckpt', but got {path}"
        param_dict = ms.load_checkpoint(path)

        # filter
        for ignore_k in ignore_keys:
            if ignore_k in param_dict:
                param_dict.pop(ignore_k)
                print(f"Deleting key {ignore_k} from state_dict.")

        ms.load_param_into_net(self, param_dict)
        print(f"Pretrain model load from '{path}' success.")

    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def encode(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    def decode(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def configure_optimizers(self) -> Any:
        raise NotImplementedError()


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        encoder_config: Dict,
        decoder_config: Dict,
        loss_config: Dict,
        regularizer_config: Dict,
        optimizer_config: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Todo: add options to freeze encoder/decoder
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.loss = instantiate_from_config(loss_config)
        self.regularization = instantiate_from_config(regularizer_config)
        self.optimizer_config = default(optimizer_config, {"target": "mindspore.nn.Adam"})
        self.lr_g_factor = lr_g_factor

    def encode(
        self, x: Tensor, return_reg_log: bool = False, unregularized: bool = False
    ) -> Union[Tensor, Tuple[Tensor, dict]]:
        z = self.encoder(x)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def construct(self, x: Tensor, **additional_decode_kwargs) -> Tuple[Tensor, Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(self, embed_dim: int, **kwargs):
        ddconfig = kwargs.pop("ddconfig")
        super().__init__(
            encoder_config={
                "target": "gm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "gm.modules.diffusionmodules.model.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        self.quant_conv = nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
            has_bias=True,
            pad_mode="valid",
        )
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1, has_bias=True, pad_mode="valid")
        self.posterior = DiagonalGaussianDistribution()
        self.embed_dim = embed_dim

    def encode(self, x: Tensor, return_reg_log: bool = False, **kwargs) -> Union[Tensor, Tuple[Tensor, dict]]:
        z = self.encoder(x)
        z = self.quant_conv(z)
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: Tensor, **decoder_kwargs) -> Tensor:
        dec = self.post_quant_conv(z)
        dec = self.decoder(dec, **decoder_kwargs)
        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        super().__init__(
            regularizer_config={"target": "gm.modules.autoencoding.regularizers.base.DiagonalGaussianRegularizer"},
            **kwargs,
        )


class AutoencoderKLModeOnly(AutoencodingEngineLegacy):
    def __init__(self, **kwargs):
        super().__init__(
            regularizer_config={
                "target": "gm.modules.autoencoding.regularizers.base.DiagonalGaussianRegularizer",
                "params": {"sample": False},
            },
            **kwargs,
        )
