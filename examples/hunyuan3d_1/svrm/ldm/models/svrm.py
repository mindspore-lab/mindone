# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import itertools
import math
import os
import shutil
import time

import cv2
import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

try:
    import mcubes
    import trimesh  # 3D repr
    import xatlas  # texture operation
except:
    raise "failed to import 3d libraries "

# load open3d, for mesh refinement
try:
    import open3d as o3d
except:
    raise "failed to import open3d library"

import inspect
from typing import Dict, Optional

from ..modules.rendering_neus.mesh import Mesh

# from ..vis_util import render
from ..util import count_params, instantiate_from_config, no_grad
from ..utils.ops import scale_tensor

# from ..modules.rendering_neus.rasterize import NVDiffRasterizerContext


def unwrap_uv(v_pos, t_pos_idx):
    print("Using xatlas to perform UV unwrapping, may take a while ...")
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_pos, t_pos_idx)
    atlas.generate(xatlas.ChartOptions(), xatlas.PackOptions())
    _, indices, uvs = atlas.get_mesh(0)
    indices = indices.astype(np.int64, casting="same_kind")
    return uvs, indices


def uv_padding(image, hole_mask, uv_padding_size=2):
    return cv2.inpaint(
        (image.asnumpy() * 255).astype(np.uint8),
        (hole_mask.asnumpy() * 255).astype(np.uint8),
        uv_padding_size,
        cv2.INPAINT_TELEA,
    )


def refine_mesh(vtx_refine, faces_refine):
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vtx_refine), triangles=o3d.utility.Vector3iVector(faces_refine)
    )

    mesh = mesh.remove_unreferenced_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()

    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound())

    mesh = mesh.simplify_vertex_clustering(
        voxel_size=0.007, contraction=o3d.geometry.SimplificationContraction.Average  # 0.005
    )

    mesh = mesh.filter_smooth_simple(number_of_iterations=2)

    vtx_refine = np.asarray(mesh.vertices).astype(np.float32)
    faces_refine = np.asarray(mesh.triangles)
    return vtx_refine, faces_refine, mesh


class SVRMModel(nn.Cell):
    def __init__(
        self, img_encoder_config, img_to_triplane_config, render_config, dtype: Optional[ms.Type] = None, **kwargs
    ):
        super(SVRMModel, self).__init__()
        self._dtype = ms.float16 if dtype is None else dtype  # inference use float16
        self.img_encoder = instantiate_from_config(img_encoder_config).to_float(
            self._dtype
        )  # FrozenDinoV2ImageEmbedder -> vision_transformer.py -> vit_base -> DinoVisionTransformer
        self.img_encoder.to(self._dtype)  # weight dtype is fp16
        self.img_to_triplane_decoder = instantiate_from_config(img_to_triplane_config).to_float(
            self._dtype
        )  # ImgToTriplaneModel
        self.img_to_triplane_decoder.to(self._dtype)  # weight dtype is fp16
        self.render = instantiate_from_config(render_config)  # TriplaneSynthesizer: inv requires float32
        self.render.to(self._dtype)  # weight dtype is fp16
        count_params(self, verbose=True)

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

    def to(self, dtype):
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, nn.Cell)]
        for module in modules:
            module.to_float(dtype)
            module.to(dtype)  # parameters
        return self

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        return expected_modules, optional_parameters

    @no_grad()
    def export_mesh_with_uv(
        self,
        data,
        mesh_size: int = 384,
        ctx=None,
        texture_res=1024,
        target_face_count=10000,
        do_texture_mapping=True,
        out_dir="outputs/test",
    ):
        """
        color_type: 0 for ray texture, 1 for vertices texture
        """

        obj_vertext_path = os.path.join(out_dir, "mesh_vertex_colors.obj")
        if do_texture_mapping:
            obj_path = os.path.join(out_dir, "mesh.obj")
            obj_texture_path = os.path.join(out_dir, "texture.png")
            obj_mtl_path = os.path.join(out_dir, "texture.mtl")
            glb_path = os.path.join(out_dir, "mesh.glb")

        st = time.time()

        here = {"dtype": ms.float16}
        input_view_image = data["input_view"].to(**here)  # [b, m, c, h, w]
        input_view_cam = data["input_view_cam"].to(**here)  # [b, m, 20]

        batch_size, input_view_num, *_ = input_view_image.shape
        assert batch_size == 1, "batch size should be 1"

        # -- encoder image
        _, _, c, h, w = input_view_image.shape
        input_view_image = input_view_image.reshape(-1, c, h, w)  # 'b m c h w -> (b m) c h w' (1*7, 3, 504, 504)
        input_view_cam = input_view_cam.reshape(-1, input_view_cam.shape[-1])  # 'b m d -> (b m) d' (1*7, 20)
        input_view_feat = self.img_encoder(input_view_image, input_view_cam)  # (7, 1297, 768)
        # '(b m) l d -> b (l m) d'
        _, l, d = input_view_feat.shape
        input_view_feat = (
            input_view_feat.reshape(-1, input_view_num, l, d).permute((0, 2, 1, 3)).reshape(-1, l * input_view_num, d)
        )

        print(f"=====> Triplane encoder time: {time.time() - st}")
        st = time.time()

        # -- decoder
        triplane_gen = self.img_to_triplane_decoder(input_view_feat)  # [b, 3, tri_dim, h, w]
        del input_view_feat

        print(f"=====> Triplane decoder time: {time.time() - st}")
        st = time.time()

        # --- triplane nerf render

        cur_triplane = triplane_gen[0:1]

        aabb = ms.Tensor([[-0.6, -0.6, -0.6], [0.6, 0.6, 0.6]]).unsqueeze(0).to(**here)
        grid_out = self.render.forward_grid(planes=cur_triplane, grid_size=mesh_size, aabb=aabb)

        # print(f"=====> Triplane forward time: {time.time() - st}")
        print(f"=====> Triplane render forward_grid time: {time.time() - st}")
        st = time.time()

        vtx, faces = mcubes.marching_cubes(0.0 - grid_out["sdf"].squeeze(0).squeeze(-1).float().asnumpy(), 0)

        bbox = aabb[0].asnumpy()
        vtx = vtx / (mesh_size - 1)
        vtx = vtx * (bbox[1] - bbox[0]) + bbox[0]

        # refine mesh
        vtx_refine, faces_refine, mesh = refine_mesh(vtx, faces)

        # reduce faces
        if faces_refine.shape[0] > target_face_count:
            print(f"reduce face: {faces_refine.shape[0]} -> {target_face_count}")
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(vtx_refine), triangles=o3d.utility.Vector3iVector(faces_refine)
            )

            # Function to simplify mesh using Quadric Error Metric Decimation by Garland and Heckbert
            mesh = mesh.simplify_quadric_decimation(target_face_count, boundary_weight=1.0)

            mesh = Mesh(
                v_pos=Tensor.from_numpy(np.asarray(mesh.vertices)),
                t_pos_idx=Tensor.from_numpy(np.asarray(mesh.triangles)),
                v_rgb=Tensor.from_numpy(np.asarray(mesh.vertex_colors)),
            )
            vtx_refine = mesh.v_pos.asnumpy()
            faces_refine = mesh.t_pos_idx.asnumpy()

        print(f"=====> refine mesh time: {time.time() - st}")
        st = time.time()

        vtx_colors = self.render.forward_points(cur_triplane, Tensor(vtx_refine).unsqueeze(0).to(**here))
        vtx_colors = vtx_colors["rgb"].float().squeeze(0).asnumpy()

        print(f"=====> generate mesh with vertex shading (Triplane forward_point) time: {time.time() - st}")
        st = time.time()

        color_ratio = 0.8  # increase brightness
        with open(obj_vertext_path, "w") as fid:
            verts = vtx_refine[:, [1, 2, 0]]
            for pidx, pp in enumerate(verts):
                color = vtx_colors[pidx]
                color = [color[0] ** color_ratio, color[1] ** color_ratio, color[2] ** color_ratio]
                fid.write("v %f %f %f %f %f %f\n" % (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
            for i, f in enumerate(faces_refine):
                f1 = f + 1
                fid.write("f %d %d %d\n" % (f1[0], f1[1], f1[2]))

        print(f"=====> generate mesh with vertex shading time: {time.time() - st}")
        st = time.time()

        if not do_texture_mapping:
            return None

        ##########  export texture  ########
        # TODO: skip texture exporting for now
        # reference: pymeshlab: https://blog.csdn.net/weixin_42605076/article/details/138429184
        # https://docs.nerf.studio/quickstart/export_geometry.html
        # https://nvlabs.github.io/nvdiffrast/#rasterization
        """
        st = time.time()

        # uv unwrap
        vtx_tex, t_tex_idx = unwrap_uv(vtx_refine, faces_refine)
        vtx_refine   = torch.from_numpy(vtx_refine).to(self.device)
        faces_refine = torch.from_numpy(faces_refine).to(self.device)
        t_tex_idx    = torch.from_numpy(t_tex_idx).to(self.device)
        uv_clip      = torch.from_numpy(vtx_tex * 2.0 - 1.0).to(self.device)

        # rasterize
        ctx = NVDiffRasterizerContext(context_type, cur_triplane.device) if ctx is None else ctx
        rast = ctx.rasterize_one(
            torch.cat([
                uv_clip,
                torch.zeros_like(uv_clip[..., 0:1]),
                torch.ones_like(uv_clip[..., 0:1])
            ], dim=-1),
            t_tex_idx,
            (texture_res, texture_res)
        )[0] # [H, W, 4]
        hole_mask = ~(rast[:, :, 3] > 0)

        # Interpolate world space position
        gb_pos = ctx.interpolate_one(vtx_refine, rast[None, ...], faces_refine)[0][0]

        with no_grad():
            gb_mask_pos_scale = scale_tensor(gb_pos.unsqueeze(0).view(1, -1, 3), (-1, 1), (-1, 1))

            tex_map = self.render.forward_points(cur_triplane, gb_mask_pos_scale)['rgb']

            tex_map = tex_map.float().squeeze(0)  # (0, 1)
            tex_map = tex_map.view((texture_res, texture_res, 3))
            img = uv_padding(tex_map, hole_mask)
            img = ((img/255.0) ** color_ratio) * 255  # increase brightness
            img = img.clip(0, 255).astype(np.uint8)

        verts = vtx_refine.cpu().numpy()[:, [1,2,0]]
        faces = faces_refine.cpu().numpy()

        with open(obj_mtl_path, 'w') as fid:
            fid.write('newmtl material_0\n')
            fid.write("Ka 1.000 1.000 1.000\n")
            fid.write("Kd 1.000 1.000 1.000\n")
            fid.write("Ks 0.000 0.000 0.000\n")
            fid.write("d 1.0\n")
            fid.write("illum 2\n")
            fid.write(f'map_Kd texture.png\n')

        with open(obj_path, 'w') as fid:
            fid.write(f'mtllib texture.mtl\n')
            for pidx, pp in enumerate(verts):
                fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
            for pidx, pp in enumerate(vtx_tex):
                fid.write('vt %f %f\n' % (pp[0], 1 - pp[1]))
            fid.write('usemtl material_0\n')
            for i, f in enumerate(faces):
                f1 = f + 1
                f2 = t_tex_idx[i] + 1
                fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2],))

        cv2.imwrite(obj_texture_path, img[..., [2, 1, 0]])
        mesh = trimesh.load_mesh(obj_path)
        mesh.export(glb_path, file_type='glb')
        print(f"=====> generate mesh with texture shading time: {time.time() - st}")
        """
