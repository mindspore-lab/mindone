from typing import Union

from mindspore import Tensor, nn
from mindspore.common import dtype as ms_dtype

from examples.stable_diffusion_v2.ldm.models.diffusion.ddpm import LatentDiffusion
from examples.t2i_adapter.adapters import StyleT2IAdapter, T2IAdapter


class SDAdapterPipeline(nn.Cell):
    """
    Wrap SD and a T2I-Adapter in a single network for easier training.

    Args:
        network: Stable Diffusion network.
        adapter: T2I adapter.
    """

    def __init__(self, network: LatentDiffusion, adapter: Union[T2IAdapter, StyleT2IAdapter]):
        super().__init__()
        self._network = network
        self._adapter = adapter

    def construct(self, x: Tensor, cond: Tensor, c: Tensor):
        """
        Args:
            x: target image.
            cond: condition image.
            c: prompt.

        Returns:
            SD Loss.
        """
        t = self._network.uniform_int(
            (x.shape[0],), Tensor(0, dtype=ms_dtype.int32), Tensor(self._network.num_timesteps, dtype=ms_dtype.int32)
        )
        x, c = self._network.get_input(x, c)
        c = self._network.get_learned_conditioning(c)
        adapter_features = self._adapter(cond)

        if isinstance(adapter_features, list):
            adapter_features = [feat.astype(self._network.dtype) for feat in adapter_features]
        else:
            adapter_features = adapter_features.astype(self._network.dtype)

        return self._network.p_losses(x, c, t, features_adapter=adapter_features)
