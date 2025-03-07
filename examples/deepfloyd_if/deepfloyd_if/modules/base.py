# -*- coding: utf-8 -*-
import os
import random
from datetime import datetime

import mindspore as ms
from mindspore import ops
from mindspore.dataset import vision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf


from .. import utils
from ..model.respace import create_gaussian_diffusion
from .utils import load_model_weights, predict_proba, clip_process_generations, get_pt2ms_mappings


class IFBaseModule:

    stage = '-'

    available_models = []
    cpu_zero_emb = np.load(os.path.join(utils.RESOURCES_ROOT, 'zero_t5-v1_1-xxl_vector.npy'))
    cpu_zero_emb = ms.Tensor(cpu_zero_emb)

    respacing_modes = {
        'fast27': '10,10,3,2,2',
        'smart27': '7,4,2,1,2,4,7',
        'smart50': '10,6,4,3,2,2,3,4,6,10',
        'smart100': '1,1,1,1,2,2,2,2,2,2,3,3,4,4,5,5,6,7,7,8,9,10,13',
        'smart185': '1,1,2,2,2,3,3,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20',
        'super27': '1,1,1,1,1,1,1,2,5,13',  # for III super-res
        'super40': '2,2,2,2,2,2,3,4,6,15',  # for III super-res
        'super100': '4,4,6,6,8,8,10,10,14,30',  # for III super-res
    }

    wm_pil_img = Image.open(os.path.join(utils.RESOURCES_ROOT, 'wm.png'))
    logo_pil_img = Image.open(os.path.join(utils.RESOURCES_ROOT, 'logo.jpeg'))

    # todo: we need to implement clip in mindspore
    # try:
    #     import clip  # noqa
    # except ModuleNotFoundError:
    #     print('Warning! You should install CLIP: "pip install git+https://github.com/openai/CLIP.git --no-deps"')
    #     raise
    #
    # clip_model, clip_preprocess = clip.load('ViT-L/14', device='cpu')
    # clip_model.eval()

    cpu_w_weights, cpu_w_biases = load_model_weights(os.path.join(utils.RESOURCES_ROOT, 'w_head_v1.npz'))
    cpu_p_weights, cpu_p_biases = load_model_weights(os.path.join(utils.RESOURCES_ROOT, 'p_head_v1.npz'))
    w_threshold, p_threshold = 0.5, 0.5

    def __init__(self, dir_or_name, pil_img_size=256, cache_dir=None, hf_token=None):
        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/IF_')
        self.dir_or_name = dir_or_name
        self.conf = self.load_conf(dir_or_name) if not self.use_diffusers else None
        self.zero_emb = self.cpu_zero_emb.copy()
        self.pil_img_size = pil_img_size

    @property
    def use_diffusers(self):
        return False

    def embeddings_to_image(
        self, t5_embs, low_res=None, *,
        style_t5_embs=None,
        positive_t5_embs=None,
        negative_t5_embs=None,
        batch_repeat=1,
        dynamic_thresholding_p=0.95,
        sample_loop='ddpm',
        sample_timestep_respacing='smart185',
        dynamic_thresholding_c=1.5,
        guidance_scale=7.0,
        aug_level=0.25,
        positive_mixer=0.15,
        blur_sigma=None,
        img_size=None,
        img_scale=4.0,
        aspect_ratio='1:1',
        progress=True,
        seed=None,
        sample_fn=None,
        support_noise=None,
        support_noise_less_qsample_steps=0,
        inpainting_mask=None,
        **kwargs,
    ):
        self._clear_cache()
        image_w, image_h = self._get_image_sizes(low_res, img_size, aspect_ratio, img_scale)
        diffusion = self.get_diffusion(sample_timestep_respacing)

        bs_scale = 2 if positive_t5_embs is None else 3

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // bs_scale]
            combined = ops.cat([half]*bs_scale, axis=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            if bs_scale == 3:
                cond_eps, pos_cond_eps, uncond_eps = ops.split(eps, len(eps) // bs_scale, axis=0)
                half_eps = uncond_eps + guidance_scale * (
                    cond_eps * (1 - positive_mixer) + pos_cond_eps * positive_mixer - uncond_eps)
                pos_half_eps = uncond_eps + guidance_scale * (pos_cond_eps - uncond_eps)
                eps = ops.cat([half_eps, pos_half_eps, half_eps], axis=0)
            else:
                cond_eps, uncond_eps = ops.split(eps, len(eps) // bs_scale, axis=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = ops.cat([half_eps, half_eps], axis=0)
            return ops.cat([eps, rest], axis=1)

        seed = self.seed_everything(seed)

        t5_embs = t5_embs.to(dtype=self.model.dtype).tile((batch_repeat, 1, 1))
        batch_size = t5_embs.shape[0] * batch_repeat

        if positive_t5_embs is not None:
            positive_t5_embs = positive_t5_embs.to(dtype=self.model.dtype).tile((batch_repeat, 1, 1))

        if negative_t5_embs is not None:
            negative_t5_embs = negative_t5_embs.to(dtype=self.model.dtype).tile((batch_repeat, 1, 1))

        timestep_text_emb = None
        if style_t5_embs is not None:
            list_timestep_text_emb = [
                style_t5_embs.to(dtype=self.model.dtype).tile((batch_repeat, 1, 1)),
            ]
            if positive_t5_embs is not None:
                list_timestep_text_emb.append(positive_t5_embs)
            if negative_t5_embs is not None:
                list_timestep_text_emb.append(negative_t5_embs)
            else:
                list_timestep_text_emb.append(
                    self.zero_emb.unsqueeze(0).tile((batch_size, 1, 1)).to(dtype=self.model.dtype))
            timestep_text_emb = ops.cat(list_timestep_text_emb, axis=0).to(dtype=self.model.dtype)

        metadata = {
            'seed': seed,
            'guidance_scale': guidance_scale,
            'dynamic_thresholding_p': dynamic_thresholding_p,
            'dynamic_thresholding_c': dynamic_thresholding_c,
            'batch_size': batch_size,
            'device_name': self.device_name,
            'img_size': [image_w, image_h],
            'sample_loop': sample_loop,
            'sample_timestep_respacing': sample_timestep_respacing,
            'stage': self.stage,
        }

        list_text_emb = [t5_embs]
        if positive_t5_embs is not None:
            list_text_emb.append(positive_t5_embs)
        if negative_t5_embs is not None:
            list_text_emb.append(negative_t5_embs)
        else:
            list_text_emb.append(
                self.zero_emb.unsqueeze(0).tile((batch_size, 1, 1)).to(dtype=self.model.dtype))

        model_kwargs = dict(
            text_emb=ops.cat(list_text_emb, axis=0).to(dtype=self.model.dtype),
            timestep_text_emb=timestep_text_emb,
            use_cache=True,
        )
        if low_res is not None:
            if blur_sigma is not None:
                low_res = vision.GaussianBlur(3, sigma=(blur_sigma, blur_sigma))(low_res)
            model_kwargs['low_res'] = ops.cat([low_res]*bs_scale, axis=0)
            model_kwargs['aug_level'] = aug_level

        if support_noise is None:
            noise = ops.randn(
                (batch_size * bs_scale, 3, image_h, image_w), dtype=self.model.dtype)
        else:
            assert support_noise_less_qsample_steps < len(diffusion.timestep_map) - 1
            assert support_noise.shape == (1, 3, image_h, image_w)
            q_sample_steps = ms.Tensor([int(len(diffusion.timestep_map) - 1 - support_noise_less_qsample_steps)])
            noise = support_noise.copy()
            noise[inpainting_mask.bool() if inpainting_mask is not None else ...] = diffusion.q_sample(
                support_noise[inpainting_mask.bool() if inpainting_mask is not None else ...],
                q_sample_steps,
            )
            noise = noise.tile((batch_size*bs_scale, 1, 1, 1)).to(dtype=self.model.dtype)

        if inpainting_mask is not None:
            inpainting_mask = inpainting_mask.to(dtype=ms.int64)

        if sample_loop == 'ddpm':
            # with torch.no_grad():
            # todo: can we safely remove no_grad?
            sample = diffusion.p_sample_loop(
                model_fn,
                (batch_size * bs_scale, 3, image_h, image_w),
                noise=noise,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                dynamic_thresholding_p=dynamic_thresholding_p,
                dynamic_thresholding_c=dynamic_thresholding_c,
                inpainting_mask=inpainting_mask,
                progress=progress,
                sample_fn=sample_fn,
            )[:batch_size]
        elif sample_loop == 'ddim':
            # with torch.no_grad():
            # todo: can we safely remove no_grad?
            sample = diffusion.ddim_sample_loop(
                model_fn,
                (batch_size * bs_scale, 3, image_h, image_w),
                noise=noise,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                dynamic_thresholding_p=dynamic_thresholding_p,
                dynamic_thresholding_c=dynamic_thresholding_c,
                progress=progress,
                sample_fn=sample_fn,
            )[:batch_size]
        else:
            raise ValueError(f'Sample loop "{sample_loop}" doesnt support')

        # todo: enable generations validation if we support clip.
        # sample = self.__validate_generations(sample)
        self._clear_cache()

        return sample, metadata

    def load_conf(self, dir_or_name, filename='config.yml'):
        path = self._get_path_or_download_file_from_hf(dir_or_name, filename)
        conf = OmegaConf.load(path)
        return conf

    def load_checkpoint(self, model, dir_or_name, filename='pytorch_model.bin'):
        path = self._get_path_or_download_file_from_hf(dir_or_name, filename)
        if os.path.exists(path):
            checkpoint_file = path
            checkpoint_file_np = f"{os.path.splitext(checkpoint_file)[0]}.npy"
            if not os.path.exists(checkpoint_file_np):
                raise FileNotFoundError(f"You need to manually transfer {checkpoint_file} to {checkpoint_file_np}")
            state_dict = np.load(checkpoint_file_np, allow_pickle=True).item()
            mappings = get_pt2ms_mappings(model)
            checkpoint = {}
            for pt_name, pt_data in state_dict.items():
                ms_name, data_mapping = mappings.get(pt_name, (pt_name, lambda x: x))
                ms_data = data_mapping(pt_data)
                checkpoint[ms_name] = ms.Parameter(ms_data.astype(np.float32), name=ms_name)
            param_not_load, ckpt_not_load = ms.load_param_into_net(model, checkpoint)
            if param_not_load or ckpt_not_load:
                print(f"{param_not_load} in network is not loaded or {ckpt_not_load} in checkpoint is not loaded!")
        else:
            print(f'Warning! In directory "{dir_or_name}" filename "pytorch_model.bin" is not found.')
        return model

    def _get_path_or_download_file_from_hf(self, dir_or_name, filename):
        if dir_or_name in self.available_models:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            print(f"You need to manually download DeepFloyd/{dir_or_name}/{filename} to {cache_dir}/{filename}!")
            return os.path.join(cache_dir, filename)
        else:
            return os.path.join(dir_or_name, filename)

    def get_diffusion(self, timestep_respacing):
        timestep_respacing = self.respacing_modes.get(timestep_respacing, timestep_respacing)
        diffusion = create_gaussian_diffusion(
            steps=1000,
            learn_sigma=True,
            sigma_small=False,
            noise_schedule='cosine',
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            timestep_respacing=timestep_respacing,
        )
        return diffusion

    @staticmethod
    def seed_everything(seed=None):
        if seed is None:
            seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        ms.set_seed(seed)
        return seed

    def device_name(self):
        return f'{ms.get_context("device_target")}_{ms.get_context("device_id")}'

    def to_images(self, generations, disable_watermark=False):
        bs, c, h, w = generations.shape
        coef = min(h / self.pil_img_size, w / self.pil_img_size)
        img_h, img_w = (int(h / coef), int(w / coef)) if coef < 1 else (h, w)

        S1, S2 = 1024 ** 2, img_w * img_h
        K = (S2 / S1) ** 0.5
        wm_size, wm_x, wm_y = int(K * 62), img_w - int(14 * K), img_h - int(14 * K)

        wm_img = self.wm_pil_img.resize(
            (wm_size, wm_size), getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None)
        logo_img = self.logo_pil_img.resize(
            (wm_size, wm_size), getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None)

        pil_images = []
        for image in ((generations + 1) * 127.5).round().clamp(0, 255).to(ms.uint8):
            pil_img = Image.fromarray(image.transpose(1, 2, 0).asnumpy()).convert('RGB')
            pil_img = pil_img.resize((img_w, img_h), getattr(Image, 'Resampling', Image).NEAREST)
            if not disable_watermark:
                pil_img.paste(wm_img, box=(wm_x - wm_size, wm_y - wm_size, wm_x, wm_y), mask=wm_img.split()[-1])
                pil_img.paste(logo_img, box=(wm_x - wm_size * 2, wm_y - wm_size, wm_x - wm_size, wm_y))
            pil_images.append(pil_img)
        return pil_images

    def show(self, pil_images, nrow=None, size=10, filename="res.jpg"):
        if nrow is None:
            nrow = round(len(pil_images)**0.5)

        def make_grid(imgs, rows, cols):
            assert len(imgs) == rows * cols

            w, h = imgs[0].size
            grid = Image.new('RGB', size=(cols * w, rows * h))
            grid_w, grid_h = grid.size

            for i, img in enumerate(imgs):
                grid.paste(img, box=(i % cols * w, i // cols * h))
            return grid

        imgs = make_grid(pil_images, nrow, len(pil_images) // nrow)
        imgs.save(filename)
        if not isinstance(imgs, list):
            imgs = [imgs]

        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(size, size))
        for i, img in enumerate(imgs):
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        fix.show()
        plt.show()

    def _clear_cache(self):
        self.model.cache = None

    def _get_image_sizes(self, low_res, img_size, aspect_ratio, img_scale):
        if low_res is not None:
            bs, c, h, w = low_res.shape
            image_h, image_w = int((h*img_scale)//32)*32, int((w*img_scale//32))*32
        else:
            scale_w, scale_h = aspect_ratio.split(':')
            scale_w, scale_h = int(scale_w), int(scale_h)
            coef = scale_w / scale_h
            image_h, image_w = img_size, img_size
            if coef >= 1:
                image_w = int(round(img_size/8 * coef) * 8)
            else:
                image_h = int(round(img_size/8 / coef) * 8)

        assert image_h % 8 == 0
        assert image_w % 8 == 0

        return image_w, image_h

    def __validate_generations(self, generations):
        # with torch.no_grad():
        # todo: can we safely remove no_grad?
        imgs = clip_process_generations(generations)
        image_features = self.clip_model.encode_image(imgs.to('cpu'))
        image_features = image_features.detach().cpu().numpy().astype(np.float16)
        p_pred = predict_proba(image_features, self.cpu_p_weights, self.cpu_p_biases)
        w_pred = predict_proba(image_features, self.cpu_w_weights, self.cpu_w_biases)
        query = p_pred > self.p_threshold
        if query.sum() > 0:
            generations[query] = vision.GaussianBlur(99, sigma=(100.0, 100.0))(generations[query])
        query = w_pred > self.w_threshold
        if query.sum() > 0:
            generations[query] = vision.GaussianBlur(99, sigma=(100.0, 100.0))(generations[query])
        return generations
