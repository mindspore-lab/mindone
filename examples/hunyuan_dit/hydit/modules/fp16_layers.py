import mindspore as ms
from mindspore import nn


class Float16Module(nn.Cell):
    def __init__(self, module, args):
        super(Float16Module, self).__init__()

        self.module = module
        self.module.to_float(ms.float16)

        self.config = self.module.config
        self.dtype = ms.float16

    def construct(
        self,
        x,
        t,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        cos_cis_img=None,
        sin_cis_img=None,
    ):
        outputs = self.module(
            x.half(),
            t.half(),
            encoder_hidden_states=encoder_hidden_states.half(),
            text_embedding_mask=text_embedding_mask.half(),
            encoder_hidden_states_t5=encoder_hidden_states_t5.half(),
            text_embedding_mask_t5=text_embedding_mask_t5.half(),
            image_meta_size=image_meta_size,
            style=style,
            cos_cis_img=cos_cis_img,
            sin_cis_img=sin_cis_img,
        )[0].float()
        return (outputs,)
