import logging
logger = logging.getLogger("canny2image")
# logger.setLevel(logging.ERROR)

import config
import os
import sys
import cv2
import einops
import numpy as np
import mindspore as ms
ms.set_seed(42)

import mindspore.ops as ops
import random
import datetime
from PIL import Image 


from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_model
from cldm.ddim_hacked import DDIMSampler
from ldm.modules.logger import set_logger


workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workspace)

def main(args):
    # set logger
    set_logger(
        name="",
        output_dir=args.output_path,
        rank=0,
        log_level=eval(args.log_level),
    )
    work_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"WORK DIR:{work_dir}")
    outpath = os.path.join(work_dir,args.output_path+args.task_name+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outpath, exist_ok=True)
    logger.info(f"Output:{outpath}")

    # set enviroment variable
    os.environ["SD_VERSION"] = "1.5"
    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(mode=ms.context.PYNATIVE_MODE, device_target='Ascend', device_id=device_id)
    # ms.set_context(mode=ms.context.GRAPH_MODE, device_target='Ascend', device_id=6)

    apply_canny = CannyDetector()

    # create model
    model = create_model('/home/mindspore/congw/project/mindone/examples/stable_diffusion_v2/configs/v1_inference_contorlnet.yaml')
    model.set_train(False)
    # load_model(model, '/home/mindspore/congw/data/ms_v1_5_pruned_emaonly-d0ab7146.ckpt')
    load_model(model, '/home/mindspore/congw/data/control_sd15_canny_ms.ckpt')

    sampler = DDIMSampler(model)

    # read image to ndarray resolution 256*256
    image_resolution = 256 # 256~768
    # input_image = np.random.randint(0, 255, (image_resolution, image_resolution, 3), dtype=np.uint8)
    image_path = args.input_image
    image = cv2.imread(image_path)
    input_image = np.array(image, dtype=np.uint8)
    
    num_samples = 4 # 1~12
    strength = 1 # 1~2
    guess_mode = False
    low_threshold =  100 #20 # 1~255
    high_threshold = 200 #100 # 1~255
    ddim_steps = 20 # 1~100
    scale = 9.0 # 0.1~30
    eta = 0.0
    # a_prompt = 'best quality, extremely detailed' 
    a_prompt = 'best quality'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality' 
    prompt = '' if args.prompt is None else args.prompt

    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape


    detected_map = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)
    Image.fromarray(detected_map).save(os.path.join(outpath, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_detected_map.png"))

    control = ms.Tensor(detected_map.copy()).float() / 255.0
    control = control.permute(2,0,1)
    control = ops.stack([control for _ in range(num_samples)], axis=0)

    
    c_crossattn = model.get_learned_conditioning(model.tokenize([prompt + ', ' + a_prompt] * num_samples))
    cond = {"c_concat": [control], "c_crossattn": [c_crossattn]}

    un_cond_c_crossattn = model.get_learned_conditioning(model.tokenize([n_prompt] * num_samples))
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [un_cond_c_crossattn]}
    
    shape = (4, H // 8, W // 8)
    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    
    logger.info('Start do inference')
    samples, intermediates = sampler.sample(ddim_steps, num_samples,
                                shape, cond, verbose=False, eta=eta,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=un_cond)

    def decode_and_save_result(samples, detected_map, outpath, filename):
        x_samples = model.decode_first_stage(samples)
        x_samples = (ops.transpose(x_samples, (0, 2, 3, 1)) * 127.5 + 127.5).asnumpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
        results = [255 - detected_map] + results


        for i, result in enumerate(results):
            img = Image.fromarray(result)
            img.save(os.path.join(outpath, f"{filename}_index{i}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png"))
            # print(result)
    

    decode_and_save_result(samples, detected_map, outpath, 'results')

    logger.info(f'Save result to {outpath} done.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--output_path", type=str, nargs="?", default="output/", help="dir to write results to")
    parser.add_argument("--input_image", type=str, help="path to input image")
    parser.add_argument("--prompt", type=str, default=None, help="prompt")  
    parser.add_argument("--task_name", type=str, default="canny2image", help="task name, used to save results")
    args = parser.parse_args()
    
    main(args)