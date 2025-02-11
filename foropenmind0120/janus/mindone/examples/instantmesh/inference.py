""" Inference of InstantMesh which uses multiview images to generate 3D mesh.

Note that the rendering part of instantmesh contains one of the cuda rasterization extensions, thus not implemented at the moment.
"""
import argparse
import os

import imageio
import mcubes
import numpy as np
from einops import rearrange
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from transformers import ViTImageProcessor
from utils.camera_util import get_sv3d_input_cameras

# from models.instantmesh.utils.mesh_util import save_obj
from utils.train_util import instantiate_from_config

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.dataset.vision import ToPIL

from mindone.utils.seed import set_random_seed


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_vid", default="INPUT_MULTIVIEW_IMG_VID_PATH", help="it has to be the 21 frames vid from sv3d output"
    )  # TODO make sure that read in video will have the exactly same nparr from the sv3d output. OTHERWISE need to dump npz from sv3d and read in.
    parser.add_argument("--name", default="anya_ms")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale of generated object.")
    parser.add_argument(
        "--config", type=str, default="models/instantmesh/configs/instant-mesh-large.yaml", help="Path to config file."
    )
    parser.add_argument("--output_path", type=str, default="outputs", help="Output directory.")
    args = parser.parse_args()
    return args


class InstantMeshPipeline(nn.Cell):
    def __init__(self, infer_config, model):
        super().__init__()
        self.infer_config = infer_config
        self.model = model

    def construct(self, inputs: ms.Tensor, radius: float) -> Tensor:
        input_cam = get_sv3d_input_cameras(radius=radius)
        logger.info(f"registered cam shape is {input_cam.shape}")
        logger.info(f"registered cam dtype is {input_cam.dtype}")
        planes = self.model.forward_planes(inputs, input_cam)
        # Uncomment this when Flexicubes available for ms
        # mesh_out = self.model.extract_mesh_with_texture(planes, **self.infer_config)
        logger.info(
            "No support for Flexicubes at the moment, due to the MS operator issues. "
            "Use a vanilla marching cube to extract meshes from SDF..."
        )
        mesh_out = self.model.extract_mesh_triplane_feat_marching_cubes(planes)
        return mesh_out


if __name__ == "__main__":
    args = args_parse()

    ms.set_context(
        mode=1,
        device_target="Ascend",
        device_id=6,
        pynative_synchronize=True,
    )
    set_random_seed(42)
    config = OmegaConf.load(args.config)
    config_name = os.path.basename(args.config).replace(".yaml", "")
    model_config = config.model_config
    infer_config = config.infer_config

    # read the vid and proc accordingly
    input_vid_arr = imageio.mimread(args.input_vid)
    images = np.asarray(input_vid_arr, dtype=np.uint8)

    logger.info("loading instantmesh model for multiview to 3d generation...")
    model = instantiate_from_config(model_config)
    model_ckpt_path = ".ckpts"
    state_dict = ms.load_checkpoint(os.path.join(model_ckpt_path, "instant_mesh_large_ms.ckpt"))
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
    m, u = ms.load_param_into_net(model, state_dict, strict_load=True)
    mesh_path = os.path.join(args.output_path, "meshes")

    # np img preprocessing
    name = args.name
    logger.info(f"Creating {name} ...")
    topil = ToPIL()
    logger.info(f"the imgae shape is {images.shape}")

    # note that ms.vision.Resizer only takes int, cannot be float
    # therefore the output clip results are slightly different from torch, which causes the triplane transformer output nan features
    # thus fc cannot have the mesh...
    # the workaround here is to use PIL built-in resize
    images = np.array([(topil(img)).resize((320, 320), Image.LANCZOS) for img in images])  # img: n h w c
    images = images.astype("float32") / 255.0
    images = images.clip(min=0, max=1)
    images = np.expand_dims(images, axis=0)  # b n h w c
    images = rearrange(images, "b n h w c -> b n c h w")  # b n c h w

    _debug_dump_vid = False
    _use_torchvision_arr = False
    if _debug_dump_vid:
        test = rearrange(images[0], "t c h w -> t h w c") * 255
        imageio.mimwrite("resized_antialias_video_jul29.mp4", test.astype(np.uint8))
    if _use_torchvision_arr:  # here load in the same image tensor as the torch version
        images = np.load("resized_np_arr.npy")  # b n c h w

    logger.info(f"the imgae shape is {images.shape}")

    # RGB image with [0,1] scale and properly sized requested by the ViTImgProc
    if images.ndim == 5:
        (B, N, C, H, W) = images.shape  # image: [B, N, C, H, W]
        images = images.reshape(B * N, C, H, W)

    # ViTImageProcessor moved out from dino wrapper to the main here, to avoid being in .construct(), do ImageNetStandard normalization
    img_processor = ViTImageProcessor.from_pretrained(model_config.params.encoder_model_name)
    inputs = img_processor(
        images=images,
        return_tensors="np",
        do_rescale=False,
        do_resize=False,
    )["pixel_values"]
    inputs = ms.Tensor(inputs).reshape(B, N, C, H, W)
    pipeline = InstantMeshPipeline(infer_config, model)
    mesh_out = pipeline(inputs, radius=4.0 * args.scale)
    mesh_path_sample = os.path.join(mesh_path, f"{name}.obj")
    # Uncomment this when Flexicubes available for ms
    # vertices, faces, vertex_colors = mesh_out
    # save_obj(
    #     vertices,
    #     faces,
    #     vertex_colors,
    #     mesh_path_sample
    # )
    sdf = mesh_out
    verts, faces = mcubes.marching_cubes(sdf.asnumpy().squeeze(0), 0)
    mcubes.export_obj(verts, faces, mesh_path_sample)
    logger.info(f"Mesh saved to {mesh_path}")
