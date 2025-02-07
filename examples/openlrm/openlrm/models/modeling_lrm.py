# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from logging import getLogger

from openlrm.utils import no_grad

import mindspore as ms
from mindspore import nn, ops

from .embedder import CameraEmbedder
from .rendering.synthesizer import TriplaneSynthesizer
from .transformer import TransformerDecoder

logger = getLogger(__name__)


class ModelLRM(nn.Cell):
    """
    Full model of the basic single-view large reconstruction model.
    """

    def __init__(
        self,
        camera_embed_dim: int,
        rendering_samples_per_ray: int,
        transformer_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        triplane_low_res: int,
        triplane_high_res: int,
        triplane_dim: int,
        encoder_freeze: bool = True,
        encoder_type: str = "dino",
        encoder_model_name: str = "facebook/dino-vitb16",
        encoder_feat_dim: int = 768,
        use_recompute: bool = False,
    ):
        super().__init__()

        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.camera_embed_dim = camera_embed_dim
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # modules
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
        )
        self.camera_embedder = CameraEmbedder(
            raw_dim=12 + 4,
            embed_dim=camera_embed_dim,
        )
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = ms.Parameter(
            ops.randn(1, 3 * triplane_low_res**2, transformer_dim) * (1.0 / transformer_dim) ** 0.5
        )
        self.transformer = TransformerDecoder(
            block_type="cond_mod",
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            mod_dim=camera_embed_dim,
        )
        self.upsampler = nn.Conv2dTranspose(
            transformer_dim, triplane_dim, kernel_size=2, stride=2, padding=0, pad_mode="valid", has_bias=True
        )
        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=triplane_dim,
            samples_per_ray=rendering_samples_per_ray,
        )

        if use_recompute:
            self.recompute(self.encoder)
            self.recompute(self.camera_embedder)
            self.recompute(self.transformer)
            self.recompute(self.upsampler)
            self.recompute(self.synthesizer)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    @property
    def dtype(self) -> ms.Type:
        r"""
        Returns:
            `mindspore.dtype`: The mindspore dtype on which the pipeline is located.
        """
        if self._dtype is not None:
            return self._dtype

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, nn.Cell)]

        for module in modules:
            self._dtype = module.weight.dtype
            return self._dtype
        self._dtype = ms.float32
        return self._dtype

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        optional_names = list(optional_parameters)
        for name in optional_names:
            if name in cls._optional_components:
                expected_modules.add(name)
                optional_parameters.remove(name)

        return expected_modules, optional_parameters

    def to(self, dtype):
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, nn.Cell)]
        for module in modules:
            module.to(dtype)
        return self

    @staticmethod
    def _encoder_fn(encoder_type: str):
        encoder_type = encoder_type.lower()
        assert encoder_type in ["dino", "dinov2"], "Unsupported encoder type"
        if encoder_type == "dino":
            from .encoders.dino_wrapper import DinoWrapper

            logger.info("Using DINO as the encoder")
            return DinoWrapper
        elif encoder_type == "dinov2":
            from .encoders.dinov2_wrapper import Dinov2Wrapper

            logger.info("Using DINOv2 as the encoder")
            return Dinov2Wrapper

    def forward_transformer(self, image_feats, camera_embeddings):
        assert (
            image_feats.shape[0] == camera_embeddings.shape[0]
        ), "Batch size mismatch for image_feats and camera_embeddings!"
        N = image_feats.shape[0]
        x = self.pos_embed.tile((N, 1, 1))  # [N, L, D]
        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )
        return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view((N, 3, H, W, -1))
        # x = ops.einsum('nihwd->indhw', x)  => # encounter error
        x = x.permute(1, 0, 4, 2, 3)  # [3, N, D, H, W]
        x = x.contiguous().view((3 * N, -1, H, W))  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view((3, N, *x.shape[-3:]))  # [3, N, D', H', W']
        # x = ops.einsum('indhw->nidhw', x)  => # encounter error
        x = x.swapaxes(0, 1)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    def forward_planes(self, image, camera):
        # image: [N, C_img, H_img, W_img]
        # camera: [N, D_cam_raw]
        N = image.shape[0]

        # encode image
        with no_grad():  # TODO: now fix encoder, may try to train encoder later with Modulated Dinov2
            image_feats = self.encoder(image)
        assert (
            image_feats.shape[-1] == self.encoder_feat_dim
        ), f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        # embed camera
        camera_embeddings = self.camera_embedder(camera)
        assert (
            camera_embeddings.shape[-1] == self.camera_embed_dim
        ), f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

        # transformer generating planes
        tokens = self.forward_transformer(image_feats, camera_embeddings)
        planes = self.reshape_upsample(tokens)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"

        return planes

    def construct(
        self,
        image,
        source_camera,
        render_cameras,
        render_anchors,
        render_resolutions,
        render_bg_colors,
        render_region_size,
    ):
        # image: [N, C_img, H_img, W_img]
        # source_camera: [N, D_cam_raw]
        # render_cameras: [N, M, D_cam_render]
        # render_anchors: [N, M, 2]
        # render_resolutions: [N, M, 1]
        # render_bg_colors: [N, M, 1]
        # render_region_size: int
        assert image.shape[0] == source_camera.shape[0], "Batch size mismatch for image and source_camera"
        assert image.shape[0] == render_cameras.shape[0], "Batch size mismatch for image and render_cameras"
        assert image.shape[0] == render_anchors.shape[0], "Batch size mismatch for image and render_anchors"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        N, M = render_cameras.shape[:2]

        planes = self.forward_planes(image, source_camera)

        # render target views
        render_results = self.synthesizer(
            planes, render_cameras, render_anchors, render_resolutions, render_bg_colors, render_region_size
        )
        assert render_results["images_rgb"].shape[0] == N, "Batch size mismatch for render_results"
        assert (
            render_results["images_rgb"].shape[1] == M
        ), "Number of rendered views should be consistent with render_cameras"

        return (
            planes,
            render_results["images_rgb"],
        )  # keys also include 'images_depth', 'images_weight' which are never used.
