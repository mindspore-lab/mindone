# reference to https://github.com/Stability-AI/generative-models
import mindspore as ms
from mindspore import Tensor, nn, ops


class IdentityWrapper(nn.Cell):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    @ms.jit
    def construct(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    @ms.jit
    def construct(
        self, x: Tensor, t: Tensor, concat: Tensor = None, context: Tensor = None, y: Tensor = None, **kwargs
    ) -> Tensor:
        if concat is not None:
            x = ops.concat((x, concat), axis=1)
        return self.diffusion_model(x, timesteps=t, context_pa=context, y=y, **kwargs)

# # # the sv3d version, does not work under ms. .get just like numpy.xx(), cannot be put under ms.Tensor.construct()?
# class OpenAIWrapper(IdentityWrapper):
#     @ms.jit
#     def construct(
#         self, x: Tensor, t: Tensor, c: dict, **kwargs
#     ) -> Tensor:
#         x = ops.concat((x, c['concat'].type_as(x)), axis=1)
#         return self.diffusion_model(x, timesteps=t, context=c['crossattn'],
#                                     y=c['vector'], **kwargs)
