from mindspore import Tensor, mint, nn


class IdentityWrapper(nn.Cell):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    # @ms.jit
    def construct(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    # @ms.jit # FIXME amp+ms.jit() err this line, also for the parent class above
    def construct(
        self, x: Tensor, t: Tensor, concat: Tensor = None, context: Tensor = None, y: Tensor = None, **kwargs
    ) -> Tensor:
        if concat is not None:
            x = mint.cat((x, concat), dim=1)
        return self.diffusion_model(x, timesteps=t, context_pa=context, y=y, **kwargs)

        # return self.diffusion_model(x, timesteps=t, context_pa=context, y=y,
        #                             image_only_indicator=mint.zeros((2, x.shape[1])),
        #                             **kwargs)
