#
# MindONE
# render and save images/gif using trimesh in CPU
# pip install pyglet
# Only support local rendering with GUI (or if remote to server by ssh -X sucessfully)
# reference: https://github.com/mikedh/trimesh/blob/main/examples/offscreen_render.py
# Usage:
# python infer/git_render.py --mesh_path mesh.obj --output_gif_path output.gif
#

import os

import imageio
import numpy as np
import trimesh
from PIL import Image
from trimesh.transformations import concatenate_matrices, rotation_matrix


def render(obj_filename, elev=0, azim=None, resolution=512, gif_dst_path="", n_views=120, fps=30, rgb=False):
    """
    obj_filename: path to obj file
    gif_dst_path:
        if set a path, will render n_views frames, then save it to a gif file
        if not set, will render single frame, then return PIL.Image instance
    rgb: if set true, will convert result to rgb image/frame
    """

    # load mesh
    mesh = trimesh.load(obj_filename)

    # get a scene object containing the mesh, this is equivalent to:
    scene = mesh.scene()

    # render scene as a PNG in bytes
    frames = []

    # rotation angles
    if azim is None:
        elev = np.linspace(elev, elev, n_views + 1)[:-1]
        azim = np.linspace(0, 360, n_views + 1)[:-1]

    xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]  # y-up direction
    scene.camera.fov = (49.1, 49.1)
    for i, (alpha, beta) in enumerate(zip(elev, azim)):
        file_name = os.path.join(os.path.dirname(gif_dst_path), "frame_%04d.png" % i)
        if i < 0 or os.path.exists(file_name):  # skip some frames if needed
            continue

        # prepare transformation matrix
        Rx = rotation_matrix(np.radians(alpha), xaxis, point=scene.centroid)
        Ry = rotation_matrix(np.radians(beta), yaxis, point=scene.centroid)
        Rz = rotation_matrix(0.0, zaxis, point=scene.centroid)
        R = concatenate_matrices(Rx, Ry, Rz)

        transform_M = scene.camera.look_at(mesh.vertices, rotation=R, distance=1.5)
        scene.camera_transform = transform_M

        while True:  # TODO: may fail after some renderings, need to be resolved
            try:
                png = scene.save_image(resolution=[resolution, resolution])  # png in bytes
                with open(file_name, "wb") as f:
                    f.write(png)
                    f.close()
                print("saved %s" % file_name)
                break
            except ZeroDivisionError:
                print("unable to save image %s" % file_name)

    if gif_dst_path != "":
        with imageio.get_writer(uri=gif_dst_path, mode="I", duration=1.0 / fps * 1000, loop=0) as writer:
            for i in range(n_views):
                file_name = os.path.join(os.path.dirname(gif_dst_path), "frame_%04d.png" % i)
                frame = Image.open(file_name).convert("RGB") if rgb else Image.open(file_name)
                writer.append_data(frame)
    return frames
