import math
import random
import mindspore as ms
from mindspore import nn, ops
# from basicsr.utils.registry import ARCH_REGISTRY

from models.gfpgan.archs.stylegan2_clean_arch import StyleGAN2GeneratorClean


class StyleGAN2GeneratorCSFT(StyleGAN2GeneratorClean):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    It is the clean version without custom compiled CUDA extensions used in StyleGAN2.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, narrow=1, sft_half=False):
        super(StyleGAN2GeneratorCSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow)
        self.sft_half = sft_half

    def construct(self,
                  styles,
                  conditions,
                  input_is_latent=False,
                  noise=None,
                  randomize_noise=True,
                  truncation=1,
                  truncation_latent=None,
                  inject_index=None,
                  return_latents=False):
        """construct function for StyleGAN2GeneratorCSFT.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}')
                         for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(
                    truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(inject_index, axis=1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(inject_index, axis=1)
            latent2 = styles[1].unsqueeze(1).repeat(
                self.num_latent - inject_index, axis=1)
            latent = ops.cat([latent1, latent2], 1)
        else:
            latent = styles[0]

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = ops.split(
                        out, int(out.shape[1] // 2), axis=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = ops.cat([out_same, out_sft], axis=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            # feature back to the rgb space
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ResBlock(nn.Cell):
    """Residual block with bilinear upsampling/downsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        mode (str): Upsampling/downsampling mode. Options: down | up. Default: down.
    """

    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.skip = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, has_bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2.0

    def construct(self, x):
        out = ops.leaky_relu(self.conv1(x), alpha=0.2)
        # upsample/downsample
        # out = ops.interpolate(out, scale_factor=self.scale_factor,
        #                       mode='bilinear', align_corners=False)
        out = ops.interpolate(out, scale_factor=self.scale_factor, mode='area')
        out = ops.leaky_relu(self.conv2(out), alpha=0.2)
        # skip
        x = ops.interpolate(x, scale_factor=self.scale_factor, mode='area')
        skip = self.skip(x)
        out = out + skip
        return out


# @ARCH_REGISTRY.register()
class GFPGANv1Clean(nn.Cell):
    """The GFPGAN architecture: Unet + StyleGAN2 decoder with SFT.

    It is the clean version without custom compiled CUDA extensions used in StyleGAN2.

    Ref: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        decoder_load_path (str): The path to the pre-trained decoder model (usually, the StyleGAN2). Default: None.
        fix_decoder (bool): Whether to fix the decoder. Default: True.

        num_mlp (int): Layer number of MLP style layers. Default: 8.
        input_is_latent (bool): Whether input is latent style. Default: False.
        different_w (bool): Whether to use different latent w for different layers. Default: False.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(
            self,
            out_size,
            num_style_feat=512,
            channel_multiplier=1,
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False):

        super(GFPGANv1Clean, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2**(int(math.log(out_size, 2)))

        self.conv_body_first = nn.Conv2d(
            3, channels[f'{first_out_size}'], kernel_size=1, has_bias=True)

        # downsample
        in_channels = channels[f'{first_out_size}']
        conv_body_down = nn.CellList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            conv_body_down.append(
                ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels

        self.conv_body_down = conv_body_down

        self.final_conv = nn.Conv2d(
            in_channels, channels['4'], kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)

        # upsample
        in_channels = channels['4']
        conv_body_up = nn.CellList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            conv_body_up.append(
                ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels

        self.conv_body_up = conv_body_up

        # to RGB
        toRGB = nn.CellList()
        for i in range(3, self.log_size + 1):
            toRGB.append(
                nn.Conv2d(channels[f'{2**i}'], 3, 1, has_bias=True))

        self.toRGB = toRGB

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2))
                                  * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = nn.Dense(
            channels['4'] * 4 * 4, linear_out_channel)

        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half)

        # load pre-trained stylegan2 model if necessary # TODO!!!
        # if decoder_load_path:
        #     self.stylegan_decoder.load_state_dict(
        #         torch.load(decoder_load_path, map_location=lambda storage, loc: storage)['params_ema'])

        # fix decoder without updating params
        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        # for SFT modulations (scale and shift)
        condition_scale = nn.CellList()
        condition_shift = nn.CellList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            condition_scale.append(
                nn.SequentialCell(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, pad_mode='pad', has_bias=True),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, sft_out_channels, kernel_size=3,
                              stride=1, padding=1, pad_mode='pad', has_bias=True)))
            condition_shift.append(
                nn.SequentialCell(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, pad_mode='pad', has_bias=True),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, sft_out_channels, kernel_size=3,
                              stride=1, padding=1, pad_mode='pad', has_bias=True)))

        self.condition_scale = condition_scale
        self.condition_shift = condition_shift

    def construct(self, x, return_latents=False, return_rgb=True, randomize_noise=True, **kwargs):
        """construct function for GFPGANv1Clean.

        Args:
            x (Tensor): Input images.
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether return intermediate rgb images. Default: True.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
        """
        conditions = []
        unet_skips = []
        out_rgbs = []

        # encoder
        feat = ops.leaky_relu(self.conv_body_first(x), alpha=0.2)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = ops.leaky_relu(self.final_conv(feat), alpha=0.2)

        # style code
        style_code = self.final_linear(feat.view(feat.shape[0], -1))
        if self.different_w:
            style_code = style_code.view(
                style_code.shape[0], -1, self.num_style_feat)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layers
            scale = self.condition_scale[i](feat)
            conditions.append(scale.copy())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.copy())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        # decoder
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)

        return image, out_rgbs
