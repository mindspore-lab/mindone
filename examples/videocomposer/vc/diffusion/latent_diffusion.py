import logging
import sys
from functools import partial

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as msnp
from mindspore import ops

sys.path.append("../stable_diffusion_v2/")
from ldm.modules.diffusionmodules.util import make_beta_schedule
from ldm.util import default, exists, extract_into_tensor

_logger = logging.getLogger(__name__)

# vc model
# class VCModelWrapper()


# net with loss + noise scheduling
class LatentDiffusion(nn.Cell):
    def __init__(
        self,
        unet: nn.Cell,
        vae: nn.Cell,  # first_stage_config
        clip_text_encoder: nn.Cell,  # cond_stage_config
        clip_image_encoder: nn.Cell = None,
        extra_conds=None,
        use_fp16=True,
        num_timesteps_cond=1,
        cond_stage_trainable=False,
        scale_factor=0.18215,  # scale vae encode z
        scale_by_std=False,
        parameterization="eps",
        timesteps=1000,
        beta_schedule="linear",
        linear_start=0.00085,
        linear_end=0.0120,
        cosine_s=8e-3,
        loss_type="l2",
        # ignore_keys=[],
        monitor="val/loss",
        given_betas=None,
        v_posterior=0.0,
        # scheduler_config=None, # not used
        learn_logvar=False,
        logvar_init=0.0,  # not used in training, used in inference sample
    ):
        """ """
        super().__init__()
        # 0. pass args
        self.cond_stage_trainable = cond_stage_trainable  # clip text encoder trainable
        if extra_conds is None:
            extra_conds = {}
        self.extra_conds = extra_conds
        if "depthmap" in extra_conds:
            self.midas = extra_conds["depthmap"]["midas"]
            self.depth_clamp = extra_conds["depthmap"]["depth_clamp"]
            self.depth_std = extra_conds["depthmap"]["depth_std"]
        if "sketch" in extra_conds or "single_sketch" in extra_conds:
            self.pidi_mean = ms.Tensor(extra_conds["sketch"]["sketch_mean"]).view(1, -1, 1, 1)
            self.pidi_std = ms.Tensor(extra_conds["sketch"]["sketch_std"]).view(1, -1, 1, 1)
            self.pidinet = extra_conds["sketch"]["pidinet"]
            self.cleaner = extra_conds["sketch"]["cleaner"]

        assert parameterization == "eps", "currently only supporting eps preiction"
        self.parameterization = parameterization
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std

        # super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        # self.model = DiffusionWrapper(unet_config, conditioning_key) # unet handled

        self.dtype = mstype.float16 if use_fp16 else mstype.float32
        self.v_posterior = v_posterior
        self.l_simple_weight = 1.0  # loss weight on mse, always 1.0 since ELBO is not used
        if monitor is not None:
            self.monitor = monitor

        # 1. create noise scheduler, i.e., betas and alphas
        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        # 2. create loss - noise prediction mse loss
        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = Tensor(np.full(shape=(self.num_timesteps,), fill_value=logvar_init).astype(np.float32))
        if self.learn_logvar:
            self.logvar = Parameter(self.logvar, requires_grad=True)
        self.randn_like = ops.StandardNormal()
        self.mse_mean = nn.MSELoss(reduction="mean")
        self.mse_none = nn.MSELoss(reduction="none")

        # 3. model register
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", Tensor(scale_factor))
        self.first_stage_model = self.vae = vae
        self.cond_text_model = self.clip_text_encoder = clip_text_encoder
        self.cond_style_model = self.clip_image_encoder = clip_image_encoder
        self.unet = unet
        if self.cond_stage_trainable:
            # unfreeze text encoder if want to finetune it
            for param in clip_text_encoder.get_parameters():
                param.requires_grad = True

        # ops
        self.uniform_int = ops.UniformInt()
        self.transpose = ops.Transpose()
        self.isnan = ops.IsNan()

        # condition select
        self.conditions = ["text", "motion", "style", "local_image"]

    # create noise scheduler
    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_mindspore = partial(Tensor, dtype=self.dtype)
        self.betas = to_mindspore(betas)
        self.alphas_cumprod = to_mindspore(alphas_cumprod)
        self.alphas_cumprod_prev = to_mindspore(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_mindspore(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_mindspore(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_mindspore(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_mindspore(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_mindspore(np.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_mindspore(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_mindspore(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = to_mindspore(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = to_mindspore((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        # self.shorten_cond_schedule = self.num_timesteps_cond > 1 # not used

    # not used in training, don's use inside construct
    # def tokenize(self, c):
    #    tokens = self.tokenizer(c, padding="max_length", max_length=77)["input_ids"]
    #    return tokens

    def get_text_embedding(self, tokens):
        c = self.cond_text_model(tokens)
        return c

    def get_image_embedding(self, image):
        pass

    def get_latent_z(self, x):
        # return self.scale_factor * z
        pass

    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    # def predict_noise(self, x_noisy, t, cond, return_ids=False, **kwargs):
    #   #     cond: it can be a dictionary or a Tensor
    #    x_recon = self.model(x_noisy, t, **cond, **kwargs)
    #    return x_recon

    def reshape_cond_data(self, cond_data, batch_size):
        # (b f) c h w -> b f c h w -> b c f h w
        cond_data = ms.ops.reshape(
            cond_data,
            (batch_size, cond_data.shape[0] // batch_size, cond_data.shape[1], cond_data.shape[2], cond_data.shape[3]),
        )
        cond_data = ms.ops.transpose(cond_data, (0, 2, 1, 3, 4))
        return cond_data

    def construct(
        self,
        x: ms.Tensor,
        text_tokens: ms.Tensor,
        fps: ms.Tensor = None,
        style_image=None,
        motion_vectors=None,
        single_image=None,  # extracted from the first frame of misc_images
        mask_seq=None,
        # depth_seq=None, # TODO: adjust to depth net inputs, containing preprocess
        # sketch_seq=None,
        # single_sketch=None, # use the first frame of sketch_seq
        misc_images=None,  # for online depth/sketch extraction, and single_image retrieval
    ):
        """
        Video composer model forward and loss computation for training

        Args:
            x: video frames, resized and normalized to shape [bs, F, 3, 256, 256]
            text: text tokens padded to fixed shape [bs, 77]
            style_image: style image (middle frame), resized and normalized to shape [bs, 3, 224, 224] in NCHW format, to clip vit-h input
            fps: frame rate, shape (1,), int
            motion_vectors: motion vectors, shape (bs, F, 2, 256, 256)
            misc_images: (bs, F, 3, 384, 384)
            single_image: first frame, resized and norm to shape [bs, 1, 3, 384, 384],
                to STC encoder. (althogh 384, but AdaptiveAvgPool2d will output fixed size features)
            depth_seq: two cases: 1) depth maps for each video frame postprocessed to shape (bs, F, 1, 384, 384).
            2)  video frames preprocessed to shape (bs, F, 3, 384, 384) for MiDas Net input.
            sketch_seq: similar
            single_sketch: sketch of first frame

        Notes:
            single_image can be derived from misc_data from dataloader
            detph_seq and sketch_seq can be extracted from misc_data for online process
        """

        # 1. sample timestep, t ~ uniform(0, T)
        # (bs, )
        t = self.uniform_int(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )

        # 2. prepare input latent frames z
        # (bs f c h w) -> (bs*f c h w) -> (bs*f z h//8 w//8) -> (b z f h//8 w//8)
        b, f, c, h_vid, w_vid = x.shape
        x = ops.reshape(x, (-1, c, h_vid, w_vid))
        # print("D--: vae input x shape", x.shape)
        z = ops.stop_gradient(self.scale_factor * self.vae.encode(x))
        z = ops.reshape(z, (b, z.shape[1], f, z.shape[2], z.shape[3]))
        # print("D--: vae output z shape: ", z.shape)

        # 3. prepare conditions

        # 3.1 text embedding
        # (bs 77) -> (bs 77 1024)
        # print("D--: clip text encoder input shape: ", text_tokens.shape)
        if self.cond_stage_trainable:
            text_emb = self.clip_text_encoder(text_tokens)
        else:
            text_emb = ops.stop_gradient(self.clip_text_encoder(text_tokens))
        # print("D--: clip text encoder output shape: ", text_emb.shape)

        # 3.2 image style embedding
        # (bs 3 224 224) -> (bs 1 1024) -> (bs 1 1 1024) -> (bs 1 1024)
        # ViT-h preprocess has been applied in dataloader
        # print("D--: style image input shape: ", style_image.shape)
        if self.clip_image_encoder is not None:
            style_emb = ops.stop_gradient(self.clip_image_encoder(style_image))
            style_emb = ops.unsqueeze(style_emb, 1)
        else:
            style_emb = None  # TODO: need to give a blank/zero value?
        # print("D--: style embedding shape: : ", style_emb.shape)

        # 3.3 motion vectors
        # (bs f 2 h w) ->  (bs 2 f h w)
        motion_vectors = ops.stop_gradient(ops.transpose(motion_vectors, (0, 2, 1, 3, 4)))
        # print("D--: motion vectors shape: : ", motion_vectors.shape)

        # 3.4 single image # TODO: change adapter to output single image
        # (bs 1 c h w) -> (bs f c h w) -> (bs c f 384 384)
        # TODO: if these tile and reshape operation is slow in MS graph, try to move to dataloader part and run with CPU.
        single_image = ops.tile(single_image, (1, f, 1, 1, 1))  # ops.unsqueeze(single_image, 1),
        single_image = ops.transpose(single_image, (0, 2, 1, 3, 4))
        # print("D--: single image shape : ", single_image.shape)

        # 3.5 fps
        #  (bs, f, 3, 384, 384) -> (bs*f, 3, 384, 384)
        misc_images = ops.reshape(misc_images, (-1, misc_images.shape[2], misc_images.shape[3], misc_images.shape[4]))
        # TODO: misc_imags are shared to use by depth and sketch extractor. \
        # the data content may change during depth extraction and the actual \
        # input for sketch extractor may not be the original misc_images! Double check the value

        # 3.6 depth:
        if "depthmap" in self.extra_conds:
            # (bs*f 3 384 384) -> (bs*f 1 384 384)
            # TODO: try chunk bs*f for 384x384 resolution and the resulting feature maps eat large memory.
            depth = ops.stop_gradient(self.midas((misc_images - 0.5) / 0.5))
            depth = ops.stop_gradient((depth / self.depth_std).clamp(0, self.depth_clamp))
            # (b*f) c h w -> b f c h w -> b c f h w
            depth = self.reshape_cond_data(depth, b)
        else:
            depth = None

        # 3.7 sketch
        if "sketch" in self.extra_conds:
            sketch = ops.stop_gradient(self.pidinet((misc_images - self.pidi_mean) / self.pidi_std))
            sketch = ops.stop_gradient(1.0 - self.cleaner(1.0 - sketch))
            # (b*f) c h w -> b f c h w -> b c f h w
            sketch = self.reshape_cond_data(sketch, b)
        else:
            sketch = None

        # 3.8 single sketch.
        # 4. diffusion forward and loss compute
        loss = self.p_losses(
            z,
            t,
            text_emb=text_emb,
            style_emb=style_emb,
            motion_vectors=motion_vectors,
            single_image=single_image,
            fps=fps,
            depth=depth,
            sketch=sketch,
        )

        return loss

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        text_emb=None,
        style_emb=None,
        motion_vectors=None,
        single_image=None,
        fps=None,  # TODO: add more conditions
        depth=None,
        sketch=None,
    ):
        # 4. add noise to latent z
        noise = msnp.randn(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 5. predict noise
        # output shape: (b c f h//8 w//8)
        noise_pred = self.unet(
            x_noisy,
            t,
            y=text_emb,
            image=style_emb,
            local_image=single_image,
            motion=motion_vectors,
            fps=fps,
            depth=depth,
            sketch=sketch,
        )

        loss_simple = self.get_loss(noise_pred, noise, mean=False).mean([1, 2, 3, 4])

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        return loss

    # TODO: try improve efficiency
    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # TODO: get loss func in init, and just call loss_func(pred, target)
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                # loss = nn.MSELoss(reduction="mean")(target, pred)
                loss = self.mse_mean(target, pred)
            else:
                # loss = nn.MSELoss(reduction="none")(target, pred)
                loss = self.mse_none(target, pred)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
