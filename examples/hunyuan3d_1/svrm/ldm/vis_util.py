# TODO
# MindONE 
# render and save images/gif using trimesh 
#

import os
from PIL import Image
import imageio
import time
import trimesh


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
    pass 

    # load mesh
    mesh = trimesh.load(obj_filename)
    
    # get a scene object containing the mesh, this is equivalent to:
    # scene = trimesh.scene.Scene(mesh)
    scene = mesh.scene()
    

    # a 45 degree homogeneous rotation matrix around
    # the Y axis at the scene centroid
    R = trimesh.transformations.rotation_matrix(
        angle=np.radians(30.0), direction=[1, 0, 0], point=scene.centroid
    )
    
    if azim is None:
        elev = np.linspace(elev, elev, n_views+1)[:-1]
        azim = np.linspace(0, 360, n_views+1)[:-1]

    # prepare R,T  then compute cameras
    corners = scene.bounds_corners # Get the bounds corners for the camera transform
    transform_M = scene.camera.look_at(corners, rotation=R)
    scene.camera_transform = transform_M
    
    fov=49.1
    # Render of scene as a PNG bytes
    png = scene.save_image()
    # Write the bytes to file
    with open("../models/featuretype.png", "wb") as f:
        f.write(png)
        f.close()
        
    R, T = look_at_view_transform(dist=1.5, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=49.1) #openGL cam

    images = renderer(meshes)

    if gif_dst_path != '': 
        with imageio.get_writer(uri=gif_dst_path, mode='I', duration=1. / fps * 1000, loop=0) as writer:
            for i in range(n_views):
                frame = images[i, ..., :3] if rgb else images[i, ...]
                frame = Image.fromarray((frame.cpu().squeeze(0) * 255).numpy().astype("uint8"))
                writer.append_data(frame)

    frame = images[..., :3] if rgb else images
    frames = [Image.fromarray((fra.cpu().squeeze(0) * 255).numpy().astype("uint8")) for fra in frame]
    return frames


