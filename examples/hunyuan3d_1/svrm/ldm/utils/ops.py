import math
from collections import defaultdict

import numpy as np

import mindspore as ms
from mindspore import mint, nn, ops

# from .typing import *
from .typing import Any, Callable, Dict, Num, Optional, Tensor, Tuple, Union

# from igl import fast_winding_number_for_meshes, point_mesh_squared_distance, read_obj
fast_winding_number_for_meshes = None
point_mesh_squared_distance = None
read_obj = None
# do not use igl yet


def dot(x, y):
    return mint.sum(x * y, dim=-1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


ValidScale = Union[Tuple[float, float], Optional[Num, Tensor]]  # Num[Tensor, "2 D"]


def scale_tensor(dat: Tensor, inp_scale: ValidScale, tgt_scale: ValidScale):
    # dat: Num[Tensor, "... D"]
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: ops.where(
            x > 0.0031308,
            ops.pow(ops.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: ops.exp(x)
    elif name == "shifted_exp":
        return lambda x: ops.exp(x - 1.0)
    # elif name == "trunc_exp":
    #     return trunc_exp
    # elif name == "shifted_trunc_exp":
    #     return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: ops.sigmoid(x)
    elif name == "tanh":
        return lambda x: ops.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: mint.nn.functional.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(ops, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


def chunk_batch(func: Callable, chunk_size: int, triplane=None, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, ms.Tensor):
            B = arg.shape[0]
            break
    assert B is not None, "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    # max(1, B) to support B == 0
    for i in range(0, max(1, B), chunk_size):
        if triplane is not None:
            out_chunk = func(
                triplane=triplane,
                *[arg[i : i + chunk_size] if isinstance(arg, ms.Tensor) else arg for arg in args],
                **{k: arg[i : i + chunk_size] if isinstance(arg, ms.Tensor) else arg for k, arg in kwargs.items()},
            )
        else:
            out_chunk = func(
                *[arg[i : i + chunk_size] if isinstance(arg, ms.Tensor) else arg for arg in args],
                **{k: arg[i : i + chunk_size] if isinstance(arg, ms.Tensor) else arg for k, arg in kwargs.items()},
            )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, ms.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f"Return value of func must be in type [ms.Tensor, list, tuple, dict], get {type(out_chunk)}.")
            exit(1)
        for k, v in out_chunk.items():
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[ms.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, ms.Tensor) for vv in v]):
            out_merged[k] = mint.cat(v, dim=0)
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, ms.Tensor)]}"
            )

    if out_type is ms.Tensor:
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
) -> Tensor:  # Float[Tensor, "H W 3"]
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

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

    i, j = ops.meshgrid(
        ops.arange(W, dtype=ms.float32) + pixel_center,
        ops.arange(H, dtype=ms.float32) + pixel_center,
        indexing="xy",
    )

    directions: Tensor = mint.stack([(i - cx) / fx, -(j - cy) / fy, -ops.ones_like(i)], -1)
    # Float[Tensor, "H W 3"]

    return directions


def get_rays(
    directions: Tensor,  # Float[Tensor, "... 3"],
    c2w: Tensor,  # Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
):  # -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].broadcast_to((rays_d.shape))
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(-1)  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].broadcast_to((rays_d.shape))
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].broadcast_to((rays_d.shape))
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].broadcast_to((rays_d.shape))

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + ops.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + ops.randn(3, device=rays_d.device) * noise_scale

    # rays_d = F.normalize(rays_d, dim=-1)
    rays_d = rays_d / ops.norm(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_projection_matrix(fovy: Tensor, aspect_wh: float, near: float, far: float) -> Tensor:
    # fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float -> Float[Tensor, "B 4 4"]
    batch_size = fovy.shape[0]
    proj_mtx = ops.zeros((batch_size, 4, 4), dtype=ms.float32)
    proj_mtx[:, 0, 0] = 1.0 / (ops.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / ops.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    # TODO: susan: please note they used nvdiffrast coord, we might flip it back if use other rasterizer

    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(c2w: Tensor, proj_mtx: Tensor) -> Tensor:
    # (c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]) -> Float[Tensor, "B 4 4"]
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    # w2c : Float[Tensor, "B 4 4"]
    w2c = ops.zeros((c2w.shape[0], 4, 4)).to(c2w.dtype)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx


def get_full_projection_matrix(c2w: Tensor, proj_mtx: Tensor) -> Tensor:
    # c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]) -> Float[Tensor, "B 4 4"]
    return (c2w.unsqueeze(0).bmm(proj_mtx.unsqueeze(0))).squeeze(0)


# gaussian splatting functions
def convert_pose(C2W):
    flip_yz = ops.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = ops.matmul(C2W, flip_yz)
    return C2W


def get_projection_matrix_gaussian(znear, zfar, fovX, fovY, device="cuda"):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = ops.zeros((4, 4))
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def get_fov_gaussian(P):
    tanHalfFovX = 1 / P[0, 0]
    tanHalfFovY = 1 / P[1, 1]
    fovY = math.atan(tanHalfFovY) * 2
    fovX = math.atan(tanHalfFovX) * 2
    return fovX, fovY


def get_cam_info_gaussian(c2w, fovx, fovy, znear, zfar):
    c2w = convert_pose(c2w)
    world_view_transform = ops.inverse(c2w)

    world_view_transform = world_view_transform.swapaxes(0, 1).float()
    projection_matrix = get_projection_matrix_gaussian(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).swapaxes(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * ops.log(input) + (1 - target) * ops.log(1 - input)).mean()
    # ops.log supports only fp16/fp32


def tet_sdf_diff(vert_sdf: Tensor, tet_edges: Tensor) -> Tensor:
    # vert_sdf: Float[Tensor, "Nv 1"], tet_edges: Integer[Tensor, "Ne 2"]) -> Float[Tensor, ""]
    sdf_f1x6x2 = vert_sdf[:, 0][tet_edges.reshape(-1)].reshape(-1, 2)
    mask = ops.sign(sdf_f1x6x2[..., 0]) != ops.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = mint.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()
    ) + mint.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


# Implementation from Latent-NeRF
# https://github.com/eladrich/latent-nerf/blob/f49ecefcd48972e69a28e3116fe95edf0fac4dc8/src/latent_nerf/models/mesh_utils.py
class MeshOBJ:
    dx = ops.zeros(3).float()
    dx[0] = 1
    dy, dz = dx[[1, 0, 2]], dx[[2, 1, 0]]
    dx, dy, dz = dx[None, :], dy[None, :], dz[None, :]

    def __init__(self, v: np.ndarray, f: np.ndarray):
        self.v = v
        self.f = f
        self.dx, self.dy, self.dz = MeshOBJ.dx, MeshOBJ.dy, MeshOBJ.dz
        self.v_tensor = ms.Tenosr(self.v)

        vf = self.v[self.f, :]
        self.f_center = vf.mean(axis=1)
        self.f_center_tensor = ms.Tenosr(self.f_center).float()

        e1 = vf[:, 1, :] - vf[:, 0, :]
        e2 = vf[:, 2, :] - vf[:, 0, :]
        self.face_normals = np.cross(e1, e2)
        self.face_normals = self.face_normals / np.linalg.norm(self.face_normals, axis=-1)[:, None]
        self.face_normals_tensor = ms.Tenosr(self.face_normals)

    def normalize_mesh(self, target_scale=0.5):
        verts = self.v

        # Compute center of bounding box
        # center = mint.mean(ops.column_stack([mint.max(verts, dim=0)[0], mint.min(verts, dim=0)[0]]))
        center = verts.mean(axis=0)
        verts = verts - center
        scale = np.max(np.linalg.norm(verts, axis=1))
        verts = (verts / scale) * target_scale

        return MeshOBJ(verts, self.f)

    def winding_number(self, query: ms.Tensor):
        shp = query.shape
        query_np = query.reshape(-1, 3).asnumpy()
        target_alphas = fast_winding_number_for_meshes(self.v.astype(np.float32), self.f, query_np)
        return ms.Tenosr(target_alphas).reshape(shp[:-1])

    def gaussian_weighted_distance(self, query: ms.Tensor, sigma):
        shp = query.shape
        query_np = query.reshape(-1, 3).asnumpy()
        distances, _, _ = point_mesh_squared_distance(query_np, self.v.astype(np.float32), self.f)
        distances = ms.Tenosr(distances).reshape(shp[:-1])
        weight = ops.exp(-(distances / (2 * sigma**2)))
        return weight


def ce_pq_loss(p, q, weight=None):
    def clamp(v, T=0.0001):
        return v.clamp(T, 1 - T)

    p = p.view((q.shape))
    ce = -1 * (p * ops.log(clamp(q)) + (1 - p) * ops.log(clamp(1 - q)))
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

    def construct(self, xyzs, sigmas):
        mesh_occ = self.sketchshape.winding_number(xyzs)
        if self.proximal_surface > 0:
            weight = 1 - self.sketchshape.gaussian_weighted_distance(xyzs, self.proximal_surface)
        else:
            weight = None
        indicator = (mesh_occ > 0.5).float()
        nerf_occ = 1 - ops.exp(-self.delta * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)
        loss = ce_pq_loss(
            nerf_occ, indicator, weight=weight
        )  # order is important for CE loss + second argument may not be optimized
        return loss


def shifted_expotional_decay(a, b, c, r):
    return a * ops.exp(-b * r) + c


def shifted_cosine_decay(a, b, c, r):
    return a * ops.cos(b * r + c) + a


def perpendicular_component(x: Tensor, y: Tensor):
    # x: Float[Tensor, "B C H W"], y: Float[Tensor, "B C H W"]
    # get the component of x that is perpendicular to y
    eps = ops.ones_like(x[:, 0, 0, 0]) * 1e-6
    return (
        x
        - (ops.mul(x, y).sum(axis=[1, 2, 3]) / ops.maximum(ops.mul(y, y).sum(axis=[1, 2, 3]), eps)).view((-1, 1, 1, 1))
        * y
    )


def validate_empty_rays(ray_indices, t_start, t_end):
    if ray_indices.nelement() == 0:
        print("Warn Empty rays_indices!")
        ray_indices = ms.Tensor([0]).to(ray_indices.dtype)
        t_start = ms.Tensor([0]).to(ray_indices.dtype)
        t_end = ms.Tensor([0]).to(ray_indices.dtype)
    return ray_indices, t_start, t_end
