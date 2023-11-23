from models.facerender.animate import AnimateFromCoeff
from mindspore import context
from inference import init_path, read_batch_from_pkl

context.set_context(mode=context.PYNATIVE_MODE,
                    device_target="Ascend", device_id=7)

sadtalker_paths = init_path('./checkpoints', './config', 'crop')
animate_from_coeff = AnimateFromCoeff(sadtalker_paths)

generator = animate_from_coeff.generator


# read_data
source_image = read_batch_from_pkl("pickles/source_image.pkl")
kp_source = read_batch_from_pkl("pickles/kp_source.pkl")
kp_norm = read_batch_from_pkl("pickles/kp_norm.pkl")

out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
