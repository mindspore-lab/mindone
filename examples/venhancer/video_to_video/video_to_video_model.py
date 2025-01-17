import logging
from typing import Any, Dict

import video_to_video.modules.unet_v2v as unet_v2v
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.modules.embedder import FrozenOpenCLIPEmbedder
from video_to_video.utils.config import cfg
from video_to_video.utils.utils import blend_time, gaussian_weights, make_chunks, pad_to_fit, sliding_windows_1d

import mindspore as ms
from mindspore import Tensor, mint, ops

from mindone.diffusers import AutoencoderKLTemporalDecoder
from mindone.utils.amp import auto_mixed_precision

logger = logging.getLogger(__name__)


class VideoToVideo:
    def __init__(self, opt):
        self.opt = opt
        clip_encoder = FrozenOpenCLIPEmbedder(pretrained="models/open_clip.ckpt")
        self.clip_encoder = clip_encoder
        logger.info(f"Build encoder with {cfg.embedder.type}")

        generator = unet_v2v.ControlledV2VUNet()
        generator.set_train(False)

        cfg.model_path = opt.model_path
        print(f"Start to load model path {cfg.model_path}")
        load_dict = ms.load_checkpoint(
            cfg.model_path,
        )
        param_not_load, _ = ms.load_param_into_net(generator, load_dict)
        print(f"Net params not loaded:{param_not_load}")
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
        )
        vae.set_train(False)

        generator = auto_mixed_precision(generator, amp_level="O2", dtype=ms.float16)
        vae = auto_mixed_precision(vae, amp_level="O2", dtype=ms.float16)
        logger.info("Use amp level O2 for generator and vae with dtype=fp16")
        self.generator = generator
        self.vae = vae

        sigmas = noise_schedule(
            schedule="logsnr_cosine_interp", n=1000, zero_terminal_snr=True, scale_min=2.0, scale_max=4.0
        )
        diffusion = GaussianDiffusion(sigmas=sigmas)
        self.diffusion = diffusion
        logger.info("Build diffusion with GaussianDiffusion")

        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt

        negative_y = clip_encoder(self.negative_prompt)
        self.negative_y = negative_y

    def test(
        self,
        input: Dict[str, Any],
        total_noise_levels=1000,
        steps=50,
        solver_mode="fast",
        guide_scale=7.5,
        noise_aug=200,
    ):
        video_data = input["video_data"]
        y = input["y"]
        mask_cond = input["mask_cond"]
        s_cond = input["s_cond"]
        interp_f_num = input["interp_f_num"]
        (target_h, target_w) = input["target_res"]

        video_data = ops.interpolate(video_data, (target_h, target_w), mode="bilinear")

        key_f_num = len(video_data)
        aug_video = []
        for i in range(key_f_num):
            if i == key_f_num - 1:
                aug_video.append(video_data[i : i + 1])
            else:
                aug_video.append(video_data[i : i + 1].tile((interp_f_num + 1, 1, 1, 1)))
        video_data = mint.cat(aug_video, dim=0)

        logger.info(f"video_data shape: {video_data.shape}")
        frames_num, _, h, w = video_data.shape

        padding = pad_to_fit(h, w)
        video_data = ops.pad(video_data, padding, "constant", 1)

        video_data = video_data.unsqueeze(0)
        mask_cond = mask_cond.unsqueeze(0)
        s_cond = Tensor(s_cond, dtype=ms.int32)

        video_data_feature = self.vae_encode(video_data)

        y = self.clip_encoder(y)

        t_hint = Tensor(noise_aug - 1, dtype=ms.int32)
        video_in_low_fps = video_data_feature[:, :, :: interp_f_num + 1]
        noised_hint = self.diffusion.diffuse(video_in_low_fps, t_hint)

        t = Tensor(total_noise_levels - 1, dtype=ms.int32)
        noised_lr = self.diffusion.diffuse(video_data_feature, t)

        model_kwargs = [{"y": y}, {"y": self.negative_y}]
        model_kwargs.append({"hint": noised_hint})
        model_kwargs.append({"mask_cond": mask_cond})
        model_kwargs.append({"s_cond": s_cond})
        model_kwargs.append({"t_hint": t_hint})

        chunk_inds = make_chunks(frames_num, interp_f_num) if frames_num > 32 else None

        solver = "heun"
        gen_vid = self.diffusion.sample(
            noise=noised_lr,
            model=self.generator,
            model_kwargs=model_kwargs,
            guide_scale=guide_scale,
            guide_rescale=0.2,
            solver=solver,
            solver_mode=solver_mode,
            steps=steps,
            t_max=total_noise_levels - 1,
            t_min=0,
            discretization="trailing",
            chunk_inds=chunk_inds,
        )

        logger.info("sampling, finished.")
        gen_video = self.tiled_chunked_decode(gen_vid)
        logger.info("temporal vae decoding, finished.")

        w1, w2, h1, h2 = padding
        gen_video = gen_video[:, :, :, h1 : h + h1, w1 : w + w1]

        return gen_video

    def temporal_vae_decode(self, z, num_f):
        return self.vae.decode(z / self.vae.config.scaling_factor, num_frames=num_f)[0]

    def vae_encode(self, t, chunk_size=1):
        bs = t.shape[0]
        # b f c h w -> (b f) c h w
        t = ops.reshape(t, (-1, *t.shape[2:]))
        z_list = []
        for ind in range(0, t.shape[0], chunk_size):
            z_list.append(self.vae.diag_gauss_dist.sample(self.vae.encode(t[ind : ind + chunk_size])[0]))
        z = mint.cat(z_list, dim=0)
        # (b f) c h w -> (b c f h w)
        z = ops.reshape(z, (bs, -1, *z.shape[1:]))
        z = ops.transpose(z, (0, 2, 1, 3, 4))
        return z * self.vae.config.scaling_factor

    def tiled_chunked_decode(self, z):
        batch_size, num_channels, num_frames, height, width = z.shape

        self.frame_chunk_size = 5
        self.tile_img_height = 576
        self.tile_img_width = 1024

        self.tile_overlap_ratio_height = 1 / 6
        self.tile_overlap_ratio_width = 1 / 5
        self.tile_overlap_ratio_time = 1 / 2

        overlap_img_height = int(self.tile_img_height * self.tile_overlap_ratio_height)
        overlap_img_width = int(self.tile_img_width * self.tile_overlap_ratio_width)

        self.tile_z_height = self.tile_img_height // 8
        self.tile_z_width = self.tile_img_width // 8

        overlap_z_height = overlap_img_height // 8
        overlap_z_width = overlap_img_width // 8

        overlap_time = int(self.frame_chunk_size * self.tile_overlap_ratio_time)

        images = z.new_zeros((batch_size, 3, num_frames, height * 8, width * 8))
        count = z.new_zeros((batch_size, 3, num_frames, height * 8, width * 8))
        height_inds = sliding_windows_1d(height, self.tile_z_height, overlap_z_height)
        for start_height, end_height in height_inds:
            width_inds = sliding_windows_1d(width, self.tile_z_width, overlap_z_width)
            for start_width, end_width in width_inds:
                time_inds = sliding_windows_1d(num_frames, self.frame_chunk_size, overlap_time)
                time = []
                for start_frame, end_frame in time_inds:
                    tile = z[
                        :,
                        :,
                        start_frame:end_frame,
                        start_height:end_height,
                        start_width:end_width,
                    ]
                    tile_f_num = tile.shape[2]
                    # b c f h w -> (b f) c h w
                    tile = ops.transpose(tile, (0, 2, 1, 3, 4))
                    tile = ops.reshape(tile, (-1, *tile.shape[2:]))
                    tile = self.temporal_vae_decode(tile, tile_f_num)

                    # (b f) c h w -> (b c f h w)
                    tile = ops.reshape(tile, (batch_size, -1, *tile.shape[1:]))
                    tile = ops.transpose(tile, (0, 2, 1, 3, 4))
                    time.append(tile)
                blended_time = []
                for k, chunk in enumerate(time):
                    if k > 0:
                        chunk = blend_time(time[k - 1], chunk, overlap_time)
                    if k != len(time) - 1:
                        chunk_size = chunk.shape[2]
                        blended_time.append(chunk[:, :, : chunk_size - overlap_time])
                    else:
                        blended_time.append(chunk)
                tile_blended_time = mint.cat(blended_time, dim=2)

                _, _, _, tile_h, tile_w = tile_blended_time.shape
                weights = gaussian_weights(tile_w, tile_h)[None, None, None]
                weights = Tensor(weights, dtype=images.dtype)

                images[:, :, :, start_height * 8 : end_height * 8, start_width * 8 : end_width * 8] += (
                    tile_blended_time * weights
                )
                count[:, :, :, start_height * 8 : end_height * 8, start_width * 8 : end_width * 8] += weights

        images = mint.div(images, count)

        return images
