""" If the extract raw mesh is large in a way, needs to scale down the color fusion part to avoid oom."""

import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import cv2
import mcubes
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

import mindspore as ms
from mindspore import mint

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../")))
import mindone.models.threestudio as threestudio
from mindone.models.threestudio.systems.base import BaseSystem
from mindone.utils.logger import set_logger
from mindone.utils.params import load_param_into_net_with_filter


def get_opts():
    parser = ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="mesh", help="scene name, used as output ply filename")
    parser.add_argument(
        "--img_wh", nargs="+", type=int, default=[512, 512], help="resolution (img_w, img_h) of the image"
    )
    parser.add_argument("--chunk", type=int, default=32 * 1024, help="chunk size to split the input to avoid OOM")
    parser.add_argument("--N_grid", type=int, default=256, help="size of the grid on 1 side, larger=higher resolution")
    parser.add_argument("--x_range", nargs="+", type=float, default=[-1.0, 1.0], help="x range of the object")
    parser.add_argument("--y_range", nargs="+", type=float, default=[-1.0, 1.0], help="x range of the object")
    parser.add_argument("--z_range", nargs="+", type=float, default=[-1.0, 1.0], help="x range of the object")
    parser.add_argument(
        "--sigma_threshold",
        type=float,
        default=30.0,  # higher threshold sparser, as the harder to be opaque
        help="threshold to consider a location is occupied",
    )
    parser.add_argument(
        "--occ_threshold",
        type=float,
        default=0.2,
        help="threshold to consider a vertex is occluded larger=fewer occluded pixels",
    )
    parser.add_argument("--use_vertex_normal", action="store_true", help="use vertex normals to compute color")
    # cfg file
    parser.add_argument("--config", default="configs/mvdream-sd21.yaml", help="path to config file")
    parser.add_argument("--resume", default="PATH/TO/ckpt/step10000.ckpt", help="path to config file")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nviews", default=60, type=int)
    parser.add_argument(
        "--fuse_color",
        action="store_true",
        help="with it can fuse color from a textureless mesh directly to generate a colored mesh,"
        "when you have a low ram device you may try this,"
        "but normally the ram bottleneck is not storing the mesh but the resolution of mesh during coloring (N_grid size).",
    )

    return parser.parse_args()


def chunk_inference(batch, renderer, chunk):
    B = batch.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = renderer(*batch)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = mint.cat(v, 0)
    return results


def render_raw_mesh(_ply_path, renderer):
    print("Predicting occupancy ...")
    with ms._no_grad():
        sigma = renderer.geometry(xyz_, output_normal=False)["density"]
    sigma = np.maximum(sigma.asnumpy(), 0).reshape(N, N, N)

    # perform marching cube algorithm to retrieve vertices and triangle mesh
    print("Extracting mesh ...")
    vertices, triangles = mcubes.marching_cubes(sigma, args.sigma_threshold)

    vertices_ = (vertices / N).astype(np.float32)
    # invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    face = np.empty(len(triangles), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = triangles

    PlyData([PlyElement.describe(vertices_[:, 0], "vertex"), PlyElement.describe(face, "face")]).write(
        f"{_ply_path}_raw.ply"
    )
    print(f"mesh to {_ply_path}")

    # remove noise in the mesh by keeping only the biggest cluster
    print("Removing noise ...")
    mesh = o3d.io.read_triangle_mesh(f"{_ply_path}_raw.ply")
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f"Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.")

    return mesh


if __name__ == "__main__":
    # setup & load model
    args = get_opts()
    output_meta_dir = "/".join(args.resume.split("/")[:-2])
    output_path = Path(os.path.join(output_meta_dir, "mesh"))
    output_path.mkdir(parents=True, exist_ok=True)
    logger = set_logger(name="", output_dir=str(output_path) if not args.debug else None)
    cfg = OmegaConf.load(args.config)
    OmegaConf.update(
        cfg, "system.guidance_type", ""
    )  # make guidance cfg empty to avoid loading mview-ldm guidance to save mem, since it's not relevant in this script
    system: BaseSystem = threestudio.find(cfg.system_type)(cfg.system, resumed=cfg.resume is not None)
    system.set_save_dir(output_path)
    system.renderer.cfg.render_to_mesh = True

    state_dict = ms.load_checkpoint(args.resume)

    # rm the 'net.' in the sd key, only do this when the system is wrapped with another net for amp
    from mindspore import Parameter

    new_sd = {}
    for k in state_dict.keys():
        #     if k.split('.')[0] in ['geometry', 'background']:
        k_list = k.split(".")
        new_k_list = k_list[1:]
        if k_list[0] == "net":
            new_sd[".".join(new_k_list)] = Parameter(state_dict[k].value())
        else:
            new_sd[k] = Parameter(state_dict[k].value())
    state_dict = new_sd

    m1, u1 = load_param_into_net_with_filter(system.renderer.geometry, state_dict)  # missing and unexpected keys
    m2, u2 = load_param_into_net_with_filter(system.renderer.background, state_dict)
    m = set(m1).union(set(m2))
    u = set(u1).intersection(set(u2))
    logger.info(f"Resumed ckpt {cfg.resume}")
    logger.info(f"missing keys {m}")
    logger.info(f"unexpected keys {u}")

    # testing: for lower nram cost
    renderer = system.renderer
    del system, state_dict, new_sd

    # not taking out the pos_emb, but just gen rays_o & rays_d from the uncond dataset
    cfg.data.n_test_views = args.nviews
    dataset = threestudio.find(cfg.data_type)(cfg.data)
    dataset.setup("mesh")
    dataset = ms.dataset.GeneratorDataset(
        dataset.test_dataset, column_names=dataset.test_dataset.output_columns, shuffle=False
    )

    # Step 1. Search for tight bounds of the obj
    # define the dense grid for query
    N = args.N_grid
    xmin, xmax = args.x_range
    ymin, ymax = args.y_range
    zmin, zmax = args.z_range
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = ms.Tensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3), dtype=ms.float32)

    # predict sigma (occupancy) for each grid location
    _ply_path = os.path.join(str(output_path), args.scene_name)

    # fuse the color in the 2nd stage (by running the script again, otherwise oom)
    if not args.fuse_color:
        mesh = render_raw_mesh(_ply_path, renderer)
    else:
        mesh = o3d.io.read_triangle_mesh(f"{_ply_path}_raw.ply")
        print("directly load raw ply for post proc")

    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)

    # perform color prediction
    # Step 0. define constants (image width, height and intrinsics) from src dataset
    W, H = dataset.source.cfg.eval_height, dataset.source.cfg.eval_width
    K = np.array([[dataset.source.focal_length, 0, W / 2], [0, dataset.source.focal_length, H / 2], [0, 0, 1]]).astype(
        np.float32
    )

    # Step 1. transform vertices into world coordinate
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1)  # (N, 4)

    # buffers to store the final averaged color
    non_occluded_sum = np.zeros((N_vertices, 1))
    v_color_sum = np.zeros((N_vertices, 3))

    # Step 2. project the vertices onto each image to infer the color
    print(f"Fusing colors with {args.nviews}...")

    rendered_img_dir = os.path.join(output_meta_dir, "outputs/save/test-it10000")

    fraction_slice_120views = int(120 / args.nviews)
    rendered_img_path = [
        os.path.join(rendered_img_dir, i)
        for i in sorted(os.listdir(rendered_img_dir), key=lambda x: int(x.split(".")[0]))
    ][
        ::fraction_slice_120views
    ]  # 30 views if 4 slices
    for view_idx, batch in tqdm(enumerate(dataset)):
        # read image of this pose
        image = Image.open(rendered_img_path[view_idx]).convert("RGB")
        image = image.resize(tuple(args.img_wh), Image.LANCZOS)
        image = np.array(image)

        # read the camera to world relative pose
        P_c2w = batch[2][0].asnumpy()  # c2w[0] as 4x4
        P_w2c = np.linalg.inv(P_c2w)[:3]  # (3, 4)

        # project vertices from world coordinate to camera coordinate
        vertices_cam = P_w2c @ vertices_homo.T  # (3, N) in "right up back"
        vertices_cam[1:] *= -1  # (3, N) in "right down forward"

        # project vertices from camera coordinate to pixel coordinate
        vertices_image = (K @ vertices_cam).T  # (N, 3)
        depth = vertices_image[:, -1:] + 1e-5  # the depth of the vertices, used as far plane
        vertices_image = vertices_image[:, :2] / depth
        vertices_image = vertices_image.astype(np.float32)
        vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W - 1)
        vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H - 1)

        # compute the color on these projected pixel coordinates
        # using bilinear interpolation.
        # NOTE: opencv's implementation has a size limit of 32768 pixels per side,
        # so may split the input into chunks.
        colors = []
        remap_chunk = int(3e4)
        for i in range(0, N_vertices, remap_chunk):
            colors += [
                cv2.remap(
                    image,
                    vertices_image[i : i + remap_chunk, 0],
                    vertices_image[i : i + remap_chunk, 1],
                    interpolation=cv2.INTER_LINEAR,
                )[:, 0]
            ]
        colors = np.vstack(colors)  # (N_vertices, 3)

        # predict occlusion of each vertex
        # we leverage the concept of NeRF by constructing rays coming out from the camera
        # and hitting each vertex; by computing the accumulated opacity along this path,
        # we can know if the vertex is occluded or not.
        # for vertices that appear to be occluded from every input view, we make the
        # assumption that its color is the same as its neighbors that are facing our side.
        # (think of a surface with one side facing us: we assume the other side has the same color)

        # each vertex's ray's origin in the raw_mesh, calculated again
        rays_o = ms.Tensor(batch[2][0][:3, -1]).expand((N_vertices, 3))

        # ray's direction is the vector pointing from camera origin to the vertices
        rays_d = ms.Tensor(vertices_) - rays_o  # (N_vertices, 3)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

        results = renderer(rays_o[None, ...], rays_d[None, ...])
        opacity = results["opacity"][0].asnumpy()  # (N_vertices, 1)
        opacity = np.nan_to_num(opacity, 1)

        non_occluded = np.ones_like(non_occluded_sum) * 0.1 / depth  # weight by inverse depth
        # near=more confident in color
        non_occluded += opacity < args.occ_threshold

        v_color_sum += colors * non_occluded
        non_occluded_sum += non_occluded

    # Step 3. combine the output and write to file
    if args.use_vertex_normal:
        v_colors = results["comp_rgb"] * 255.0
    else:  # the combined color is the average color among all views
        v_colors = v_color_sum / non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertices_.dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr + v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]

    face = np.empty(len(triangles), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = triangles

    PlyData([PlyElement.describe(vertex_all, "vertex"), PlyElement.describe(face, "face")]).write(
        f"{_ply_path}_color.ply"
    )

    print("Color Mesh Extraction Done!")
