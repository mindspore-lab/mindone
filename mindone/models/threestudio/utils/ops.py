"""for those used in dataset ops, are implemented as np functions"""

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import threestudio
from igl import fast_winding_number_for_meshes, point_mesh_squared_distance, read_obj

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Tensor, mint, nn


def dot(x, y):
    return mint.sum(x * y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


ValidScale = Union[Tuple[float, float]]


def scale_tensor(dat: Tensor, inp_scale: ValidScale, tgt_scale: ValidScale):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


class SpecifyGradient:
    # Implementation from stable-dreamfusion
    # https://github.com/ashawkey/stable-dreamfusion
    @staticmethod
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return mint.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: mint.where(
            x > 0.0031308,
            mint.pow(mint.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: mint.exp(x)
    elif name == "shifted_exp":
        return lambda x: mint.exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: mint.sigmoid(x)
    elif name == "tanh":
        return lambda x: mint.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, mint.Tensor):
            B = arg.shape[0]
            break
    assert B is not None, "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    # max(1, B) to support B == 0
    for i in range(0, max(1, B), chunk_size):
        out_chunk = func(
            *[arg[i : i + chunk_size] if isinstance(arg, mint.Tensor) else arg for arg in args],
            **{k: arg[i : i + chunk_size] if isinstance(arg, mint.Tensor) else arg for k, arg in kwargs.items()},
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, mint.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f"Return value of func must be in type [mint.Tensor, list, tuple, dict], get {type(out_chunk)}.")
            exit(1)
        for k, v in out_chunk.items():
            v = v if mint.is_grad_enabled() else v
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[mint.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, mint.Tensor) for vv in v]):
            out_merged[k] = mint.cat(v, dim=0)
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, mint.Tensor)]}"
            )

    if out_type is mint.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> np.array:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing="xy",
    )

    directions: np.array = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], -1)

    return directions


def l2norm_np(x: np.array, axis=-1, permute_dims=False):
    l2norm = np.linalg.norm(x, ord=2, axis=axis)
    if not permute_dims:
        return x / np.repeat(l2norm[:, None], x.shape[axis], axis=1)
    else:
        return x / np.transpose(
            np.repeat(l2norm[:, None], x.shape[axis], axis=1), (0, 2, 3, 1)
        )  # (8,3,64,64) -> (8,64,64,3)


def get_rays(
    directions: np.array,
    c2w: np.array,
    keepdim=False,
    noise_scale=0.0,
) -> Tuple[np.array, np.array]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = np.broadcast_to(c2w[:, :3, 3], rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(-1)  # (H, W, 3)
            rays_o = np.broadcast_to(c2w[None, None, :3, 3], rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)  # (B, H, W, 3)
            rays_o = np.broadcast_to(c2w[:, None, None, :3, 3], rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)  # (B, H, W, 3)
        rays_o = np.broadcast_to(c2w[:, None, None, :3, 3], rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + np.random.randn(3) * noise_scale
        rays_d = rays_d + np.random.randn(3) * noise_scale

    rays_d = l2norm_np(rays_d, permute_dims=True)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_projection_matrix(fovy: np.array, aspect_wh: float, near: float, far: float) -> np.array:
    batch_size = fovy.shape[0]
    proj_mtx = np.zeros(shape=(batch_size, 4, 4), dtype=np.float32)
    proj_mtx[:, 0, 0] = 1.0 / (np.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / np.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(c2w: np.array, proj_mtx: np.array) -> np.array:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c: np.array = np.zeros(shape=(c2w.shape[0], 4, 4)).astype(c2w.dtype)
    w2c[:, :3, :3] = np.transpose(c2w[:, :3, :3], (0, 2, 1))
    w2c[:, :3, 3:] = np.transpose(-c2w[:, :3, :3], (0, 2, 1)) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * mint.log(input) + (1 - target) * mint.log(1 - input)).mean()


def tet_sdf_diff(vert_sdf: Tensor, tet_edges: Tensor) -> Tensor:
    sdf_f1x6x2 = vert_sdf[:, 0][tet_edges.reshape(-1)].reshape(-1, 2)
    mask = mint.sign(sdf_f1x6x2[..., 0]) != mint.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()
    ) + F.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


# Implementation from Latent-NeRF
# https://github.com/eladrich/latent-nerf/blob/f49ecefcd48972e69a28e3116fe95edf0fac4dc8/src/latent_nerf/models/mesh_utils.py
class MeshOBJ:
    dx = np.zeros(3, dtype=np.float32)
    dx[0] = 1
    dy, dz = dx[[1, 0, 2]], dx[[2, 1, 0]]
    dx, dy, dz = dx[None, :], dy[None, :], dz[None, :]

    def __init__(self, v: np.ndarray, f: np.ndarray):
        self.v = v
        self.f = f
        self.dx, self.dy, self.dz = MeshOBJ.dx, MeshOBJ.dy, MeshOBJ.dz
        self.v_tensor = Tensor(self.v)

        vf = self.v[self.f, :]
        self.f_center = vf.mean(axis=1)
        self.f_center_tensor = Tensor(self.f_center).float()

        e1 = vf[:, 1, :] - vf[:, 0, :]
        e2 = vf[:, 2, :] - vf[:, 0, :]
        self.face_normals = np.cross(e1, e2)
        self.face_normals = self.face_normals / np.linalg.norm(self.face_normals, axis=-1)[:, None]
        self.face_normals_tensor = Tensor(self.face_normals)

    def normalize_mesh(self, target_scale=0.5):
        verts = self.v

        # Compute center of bounding box
        # center = mint.mean(mint.column_stack([mint.max(verts, dim=0)[0], mint.min(verts, dim=0)[0]]))
        center = verts.mean(axis=0)
        verts = verts - center
        scale = np.max(np.linalg.norm(verts, axis=1))
        verts = (verts / scale) * target_scale

        return MeshOBJ(verts, self.f)

    def winding_number(self, query: mint.Tensor):
        shp = query.shape
        query_np = query.asnumpy()
        target_alphas = fast_winding_number_for_meshes(self.v.astype(np.float32), self.f, query_np)
        return Tensor(target_alphas).reshape(shp[:-1])

    def gaussian_weighted_distance(self, query: mint.Tensor, sigma):
        shp = query.shape
        query_np = query.asnumpy()
        distances, _, _ = point_mesh_squared_distance(query_np, self.v.astype(np.float32), self.f)
        distances = Tensor(distances).reshape(shp[:-1])
        weight = mint.exp(-(distances / (2 * sigma**2)))
        return weight


def ce_pq_loss(p, q, weight=None):
    def clamp(v, T=0.0001):
        return v.clamp(T, 1 - T)

    p = p.view(q.shape)
    ce = -1 * (p * mint.log(clamp(q)) + (1 - p) * mint.log(clamp(1 - q)))
    if weight is not None:
        ce *= weight
    return ce.sum()


class ShapeLoss(nn.Cell):
    def __init__(self, guide_shape):
        super().__init__()
        self.mesh_scale = 0.7
        self.proximal_surface = 0.3
        self.delta = 0.2
        self.shape_path = guide_shape
        v, _, _, f, _, _ = read_obj(self.shape_path, float)
        mesh = MeshOBJ(v, f)
        matrix_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        self.sketchshape = mesh.normalize_mesh(self.mesh_scale)
        self.sketchshape = MeshOBJ(
            np.ascontiguousarray((matrix_rot @ self.sketchshape.v.transpose(1, 0)).transpose(1, 0)),
            f,
        )

    def forward(self, xyzs, sigmas):
        mesh_occ = self.sketchshape.winding_number(xyzs)
        if self.proximal_surface > 0:
            weight = 1 - self.sketchshape.gaussian_weighted_distance(xyzs, self.proximal_surface)
        else:
            weight = None
        indicator = (mesh_occ > 0.5).float()
        nerf_occ = 1 - mint.exp(-self.delta * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)
        loss = ce_pq_loss(
            nerf_occ, indicator, weight=weight
        )  # order is important for CE loss + second argument may not be optimized
        return loss


def shifted_expotional_decay(a, b, c, r):
    return a * mint.exp(-b * r) + c


def shifted_cosine_decay(a, b, c, r):
    return a * mint.cos(b * r + c) + a


def perpendicular_component(x: Tensor, y: Tensor):
    # get the component of x that is perpendicular to y
    eps = mint.ones_like(x[:, 0, 0, 0]) * 1e-6
    return (
        x
        - (mint.mul(x, y).sum(dim=[1, 2, 3]) / mint.maximum(mint.mul(y, y).sum(dim=[1, 2, 3]), eps)).view(-1, 1, 1, 1)
        * y
    )


def validate_empty_rays(ray_indices, t_start, t_end):
    if ray_indices.nelement() == 0:
        threestudio.info("Empty rays_indices!")
        ray_indices = Tensor([0], dtype=ms.int32).to(ray_indices)
        t_start = mint.Tensor([0]).to(ray_indices)
        t_end = mint.Tensor([0]).to(ray_indices)
    return ray_indices, t_start, t_end
