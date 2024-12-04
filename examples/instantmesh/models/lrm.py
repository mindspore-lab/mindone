from typing import Optional, Tuple

import mcubes

import mindspore as ms
import mindspore.nn as nn
from mindspore import mint

from .decoder.transformer import TriplaneTransformer
from .encoder.dino_wrapper import DinoWrapper
from .renderer.synthesizer import TriplaneSynthesizer


class InstantNeRF(nn.Cell):
    """
    Full model for training the LRM with nerf.
    """

    def __init__(
        self,
        encoder_freeze: bool = False,
        encoder_model_name: str = "facebook/dino-vitb16",
        encoder_feat_dim: int = 768,
        transformer_dim: int = 1024,
        transformer_layers: int = 16,
        transformer_heads: int = 16,
        triplane_low_res: int = 32,
        triplane_high_res: int = 64,
        triplane_dim: int = 80,
        rendering_samples_per_ray: int = 128,
        render_size: int = 192,
        use_recompute: bool = False,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.render_size = render_size
        self.chunk_size = 1

        # modules
        self.encoder = DinoWrapper(
            model_name=encoder_model_name,
            use_recompute=use_recompute,  # enable the finest recompute
        )

        dtype_map = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}
        dtype = dtype_map[dtype]
        self.transformer = TriplaneTransformer(
            inner_dim=transformer_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            image_feat_dim=encoder_feat_dim,
            triplane_low_res=triplane_low_res,
            triplane_high_res=triplane_high_res,
            triplane_dim=triplane_dim,
            dtype=dtype,
            use_recompute=use_recompute,  # enable the finest recompute
        )

        self.synthesizer = TriplaneSynthesizer(
            triplane_dim=triplane_dim,
            samples_per_ray=rendering_samples_per_ray,
            dtype=dtype,
            use_recompute=use_recompute,  # enable the finest recompute
        )

    def forward_planes(self, images: ms.Tensor, cameras: ms.Tensor):
        # cameras: b n 16
        # images: b n c h w
        B = images.shape[0]

        # encode images
        image_feats = self.encoder(images, cameras)

        image_feats = image_feats.reshape(
            B, int(image_feats.shape[-3] * image_feats.shape[-2] / B), image_feats.shape[-1]
        )
        # logger.info(f'the shape in forward plane after reshape: {image_feats.shape}')

        # transformer decode the plane feat
        planes = self.transformer(image_feats)

        return planes

    def construct(
        self,
        images: ms.Tensor,
        cameras: ms.Tensor,
        render_cameras: ms.Tensor,
        render_size: int,
        crop_params: Optional[Tuple[int]],
    ):
        # images: [B, V, C_img, H_img, W_img]
        # cameras: [B, V, 16]
        # render_cameras: [B, M, D_cam_render]
        B, V = render_cameras.shape[:2]

        planes = self.forward_planes(images, cameras)

        l_rgb_depth_weight = []
        for i in range(0, V, self.chunk_size):
            syn_out = self.synthesizer(
                planes,
                cameras=render_cameras[:, i : i + self.chunk_size],
                render_size=render_size,
                crop_params=crop_params,
            )
            l_rgb_depth_weight.append(syn_out)
        images_rgb = mint.cat([view[0] for view in l_rgb_depth_weight], dim=1)
        images_depth = mint.cat([view[1] for view in l_rgb_depth_weight], dim=1)
        images_weight = mint.cat([view[2] for view in l_rgb_depth_weight], dim=1)

        return images_rgb, images_depth, images_weight

    def get_texture_prediction(self, planes, tex_pos, hard_mask=None):
        """
        Predict Texture given triplanes
        :param planes: the triplane feature map
        :param tex_pos: Position we want to query the texture field
        :param hard_mask: 2D silhoueete of the rendered image
        """
        tex_pos = mint.cat(tex_pos, dim=0)
        if hard_mask is not None:
            tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)
        ###################
        # We use mask to get the texture location (to save the memory)
        if hard_mask is not None:
            n_point_list = mint.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1)
            sample_tex_pose_list = []
            max_point = n_point_list.max()
            expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5
            for i in range(tex_pos.shape[0]):
                tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3)
                if tex_pos_one_shape.shape[1] < max_point:
                    tex_pos_one_shape = mint.cat(
                        [
                            tex_pos_one_shape,
                            mint.zeros((1, max_point - tex_pos_one_shape.shape[1], 3), dtype=ms.float32),
                        ],
                        dim=1,
                    )
                sample_tex_pose_list.append(tex_pos_one_shape)
            tex_pos = mint.cat(sample_tex_pose_list, dim=0)

        tex_feat = self.synthesizer.forward_points(
            planes,
            tex_pos,
        )["rgb"]

        if hard_mask is not None:
            final_tex_feat = mint.zeros(planes.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1])
            expanded_hard_mask = (
                hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            )
            for i in range(planes.shape[0]):
                final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][: n_point_list[i]].reshape(-1)
            tex_feat = final_tex_feat

        return tex_feat.reshape(planes.shape[0], hard_mask.shape[1], hard_mask.shape[2], tex_feat.shape[-1])

    def extract_mesh(
        self,
        planes: ms.Tensor,
        mesh_resolution: int = 256,
        mesh_threshold: int = 10.0,
        texture_resolution: int = 1024,
    ):
        """
        Extract a 3D mesh from triplane nerf. Only support batch_size 1.
        :param planes: triplane features
        :param mesh_resolution: marching cubes resolution
        :param mesh_threshold: iso-surface threshold
        :param use_texture_map: use texture map or vertex color
        :param texture_resolution: the resolution of texture map
        """
        assert planes.shape[0] == 1

        grid_out = self.synthesizer.forward_grid(
            planes=planes,
            grid_size=mesh_resolution,
        )

        vertices, faces = mcubes.marching_cubes(
            grid_out["sigma"].squeeze(0).squeeze(-1).cpu().numpy(),
            mesh_threshold,
        )
        vertices = vertices / (mesh_resolution - 1) * 2 - 1

        # query vertex colors
        vertices_tensor = ms.tensor(vertices, dtype=ms.float32).unsqueeze(0)
        vertices_colors = self.synthesizer.forward_points(planes, vertices_tensor)["rgb"].squeeze(0)
        vertices_colors = (vertices_colors * 255).to(ms.uint8)

        return vertices, faces, vertices_colors
