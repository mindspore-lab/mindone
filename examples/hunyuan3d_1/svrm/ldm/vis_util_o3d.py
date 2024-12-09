# 
# MindONE 
# render and save images/gif using open3d in CPU
# pip install pyglet
# Only support local rendering with GUI (or if remote to server by ssh -X sucessfully)
# reference: https://github.com/isl-org/Open3D/blob/main/examples/python/visualization/render_to_image.py
# Usage:
# python infer/git_render.py --mesh_path mesh.obj --output_gif_path output.gif
#

import os
os.environ['EGL_PLATFORM'] = 'surfaceless'   # Ubuntu 20.04+

from PIL import Image
import imageio
import time
import numpy as np
import open3d as o3d
from open3d.visualization import rendering


def render(
    obj_filename, 
    elev=0, 
    azim=None, 
    resolution=512, 
    gif_dst_path='', 
    n_views=120, 
    fps=30, 
    rgb=False
):
    '''
        obj_filename: path to obj file
        gif_dst_path: 
            if set a path, will render n_views frames, then save it to a gif file
            if not set, will render single frame, then return PIL.Image instance
        rgb: if set true, will convert result to rgb image/frame
    '''
    
    # load mesh
    mesh = o3d.io.read_triangle_model(obj_filename)

    # offscreen renderer configs
    w, h = resolution, resolution
    render = rendering.OffscreenRenderer(w, h)
    render.scene.add_model("mesh", mesh)
    render.setup_camera(vertical_field_of_view = 49.1)
    render.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
    render.scene.scene.enable_sun_light(True)
    # render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))    
    
    # render scene
    frames = []

    # rotation angles
    if azim is None:
        elev = np.linspace(elev, elev, n_views+1)[:-1]
        azim = np.linspace(0, 360, n_views+1)[:-1]


    xaxis, yaxis, zaxis = [1,0,0], [0,1,0], [0,0,1] # y-up direction
    scene.camera.fov = (49.1, 49.1)
    for i, (alpha, beta) in enumerate(zip(elev, azim)):
        if i < 0: #skip some frames if needed 
            continue
             
        aspect = h/w
        s = 3
        
        # In the code below we rotate the mesh using Euler angles.
        # https://www.open3d.org/docs/latest/python_api/open3d.geometry.TriangleMesh.html#open3d.geometry.TriangleMesh.rotate
        # https://www.open3d.org/docs/latest/tutorial/geometry/transformation.html#Rotation
        mesh_r = copy.deepcopy(mesh)
        R = mesh.get_rotation_matrix_from_xyz((np.radians(alpha), np.radians(beta), 0.)) # right-x, up-y, backward-z
        mesh_r.rotate(R, center=mesh_r.get_center())


        render.scene.camera.set_projection(field_of_view=49.1, 
                                            aspect_ratio=aspect, 
                                            near=0., far=100.,
                                            field_of_view_type=rendering.Camera.FovType.Vertical)
        # set_projection(self: open3d.cpu.pybind.visualization.rendering.Camera, 
        # field_of_view: float, aspect_ratio: float, near_plane: float, far_plane: float, 
        # field_of_view_type: open3d.cpu.pybind.visualization.rendering.Camera.FovType) -> None

        # render.scene.camera.set_projection(rendering.Camera.Projection.Ortho,
        #                                     -s, s, -s*aspect, s*aspect, 0.1, 200)
        # set_projection(self: open3d.cpu.pybind.visualization.rendering.Camera, 
        # projection_type: open3d.cpu.pybind.visualization.rendering.Camera.Projection, 
        # left: float, right: float, bottom: float, top: float, near: float, far: float) -> None
        render.scene.camera.look_at(center=mesh.get_center(), eye=mesh.get_center+[0., 0., 1.5], up=[0, 1, 0])

        file_name = os.path.join(os.path.dirname(gif_dst_path), "frame_%04d.png"%i)
        img = render.render_to_image()
        o3d.io.write_image(file_name, img)
        print("saved %s"%file_name)

    if gif_dst_path != '': 
        with imageio.get_writer(uri=gif_dst_path, mode='I', duration=1. / fps * 1000, loop=0) as writer:
            for i in range(n_views):
                file_name = os.path.join(os.path.dirname(gif_dst_path), "frame_%04d.png"%i)
                frame = Image.open(file_name).convert("RGB") if rgb else Image.open(file_name)
                writer.append_data(frame)
    return frames


