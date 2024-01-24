import pickle
import mindspore as ms
from mindspore import context


def param_convert(ms_params, pt_params, ckpt_path, extra_dict=None):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}

    if extra_dict:
        bn_ms2pt.update(extra_dict)

    new_params_list = []
    for ms_param in ms_params:
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数

        # if "conv_block.1." in ms_param.name:
        # if any(x in ms_param.name and "mlp_gamma" not in ms_param.name and "mlp_beta" not in ms_param.name for x in bn_ms2pt.keys()):
        if True:
            # ms_param_item = ms_param.name.split(".")
            # pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            # pt_param = ".".join(pt_param_item)

            param_name = ms_param.name
            for k, v in bn_ms2pt.items():
                param_name = param_name.replace(k, v)
            pt_param = param_name

            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_param.data.shape:
                ms_value = pt_params[pt_param]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32)})
            elif pt_param in pt_params and "weight" in ms_param.name:
                ms_value = pt_params[pt_param]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32).unsqueeze(2)})
            else:
                print(ms_param.name, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            if ms_param.name in pt_params and tuple(pt_params[ms_param.name].shape) == tuple(ms_param.data.shape):
                ms_value = pt_params[ms_param.name]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32)})
            # elif ms_param.name in pt_params and ("weight_u" in ms_param.name or "weight_v" in ms_param.name):
            elif ms_param.name in pt_params and "weight" in ms_param.name:
                ms_value = pt_params[ms_param.name]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32).unsqueeze(2)})
            else:
                print(ms_param.name, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)


def convert_audio2exp():
    from models.audio2exp.expnet import ExpNet
    from models.audio2exp.audio2exp import Audio2Exp
    from yacs.config import CfgNode as CN

    fcfg_exp = open("config/audio2exp.yaml")
    cfg_exp = CN.load_cfg(fcfg_exp)
    cfg_exp.freeze()
    fcfg_exp.close()

    netG = ExpNet()
    audio2exp_model = Audio2Exp(netG, cfg_exp, prepare_training_loss=False)
    ms_params = audio2exp_model.get_parameters()

    with open("../SadTalker/pt_weights_audio2exp.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict, "checkpoints/ms/ms_audio2exp.ckpt")


def convert_audio2pose():

    from models.audio2pose.audio2pose import Audio2Pose
    from yacs.config import CfgNode as CN

    fcfg_pose = open("config/audio2pose.yaml")
    cfg_pose = CN.load_cfg(fcfg_pose)
    cfg_pose.freeze()
    fcfg_pose.close()

    extra_dict = {
        "mlp.0": "MLP.L0",
        "mlp.2": "MLP.L1",
    }

    audio2exp_model = Audio2Pose(cfg_pose)
    ms_params = audio2exp_model.get_parameters()

    with open("../SadTalker/pt_weights_audio2pose.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_audio2pose.ckpt", extra_dict)


def convert_netrecon():
    from models.face3d.networks import define_net_recon

    net = define_net_recon(net_recon='resnet50',
                           use_last_fc=False, init_path='')
    ms_params = net.get_parameters()

    with open("../SadTalker/pt_weights_net_recon.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict, "checkpoints/ms/ms_net_recon.ckpt")


def convert_retinaface():
    from models.face3d.facexlib import init_detection_model

    det_net = init_detection_model('retinaface_resnet50', half=False)
    ms_params = det_net.get_parameters()

    with open("../SadTalker/pt_weights_retinaface.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_det_retinaface.ckpt")


def convert_awingfan():
    from models.face3d.facexlib import init_alignment_model
    detector = init_alignment_model('awing_fan')
    ms_params = detector.get_parameters()

    with open("../SadTalker/pt_weights_fan.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_detector_fan.ckpt")


def convert_mapping():

    import yaml
    from models.facerender.modules.mapping import MappingNet

    with open('config/facerender_still.yaml') as f:
        config = yaml.safe_load(f)

    mapping = MappingNet(**config['model_params']['mapping_params'])

    ms_params = mapping.get_parameters()

    with open("pickles/pt_weights_mapping_full.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_mapping_full.ckpt")


def convert_generator():
    import yaml
    from models.facerender.modules.generator import OcclusionAwareSPADEGenerator

    with open('config/facerender.yaml') as f:
        config = yaml.safe_load(f)

    ms2pt = {
        'resblocks_3d.0.': 'resblocks_3d.3dr0.',
        'resblocks_3d.1.': 'resblocks_3d.3dr1.',
        'resblocks_3d.2.': 'resblocks_3d.3dr2.',
        'resblocks_3d.3.': 'resblocks_3d.3dr3.',
        'resblocks_3d.4.': 'resblocks_3d.3dr4.',
        'resblocks_3d.5.': 'resblocks_3d.3dr5.',
        ".bn2d": "",
    }

    generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                             **config['model_params']['common_params'])

    ms_params = generator.get_parameters()

    with open("../SadTalker/pt_weights_generator.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_generator.ckpt", ms2pt)


def convert_heestimator():
    import yaml
    from models.facerender.modules.keypoint_detector import HEEstimator

    with open('config/facerender.yaml') as f:
        config = yaml.safe_load(f)

    ms2pt = {
        "block1.0.": "block1.b1_0.",
        "block1.1.": "block1.b1_1.",
        "block1.2.": "block1.b1_2.",
        "block3.0.": "block3.b3_0.",
        "block3.1.": "block3.b3_1.",
        "block3.2.": "block3.b3_2.",
        "block5.0.": "block5.b5_0.",
        "block5.1.": "block5.b5_1.",
        "block5.2.": "block5.b5_2.",
        "block5.3.": "block5.b5_3.",
        "block5.4.": "block5.b5_4.",
        "block7.0.": "block7.b7_0.",
        "block7.1.": "block7.b7_1.",
    }

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])

    ms_params = he_estimator.get_parameters()

    with open("../SadTalker/pt_weights_he_estimator.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_he_estimator.ckpt", ms2pt)


def convert_kpextractor():
    import yaml
    from models.facerender.modules.keypoint_detector import KPDetector

    with open('config/facerender.yaml') as f:
        config = yaml.safe_load(f)

    ms2pt = {
        "predictor.down_blocks.0": "predictor.down_blocks.down0",
        "predictor.down_blocks.1": "predictor.down_blocks.down1",
        "predictor.down_blocks.2": "predictor.down_blocks.down2",
        "predictor.down_blocks.3": "predictor.down_blocks.down3",
        "predictor.down_blocks.4": "predictor.down_blocks.down4",
        "predictor.up_blocks.0": "predictor.up_blocks.up0",
        "predictor.up_blocks.1": "predictor.up_blocks.up1",
        "predictor.up_blocks.2": "predictor.up_blocks.up2",
        "predictor.up_blocks.3": "predictor.up_blocks.up3",
        "predictor.up_blocks.4": "predictor.up_blocks.up4",
        ".bn2d": "",
    }

    kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                              **config['model_params']['common_params'])

    ms_params = kp_extractor.get_parameters()

    with open("../SadTalker/pt_weights_kp_extractor.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_kp_extractor.ckpt", ms2pt)


def convert_gfpgan():
    from models.gfpgan.gfpganer import GFPGANer

    model_path = None
    arch = 'clean'
    channel_multiplier = 2
    bg_upsampler = None

    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    net = restorer.gfpgan

    ms_params = net.get_parameters()
    with open("pickles/pt_weights_gfpgan.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_gfpgan.ckpt")


def convert_parsing():
    from models.face3d.facexlib.parsenet import ParseNet
    model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
    ms_params = model.get_parameters()
    with open("pickles/pt_weights_parsing.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "gfpgan/weights/parsing_parsenet.ckpt")


def convert_wav2lip():
    from models.audio2exp.wav2lip import Wav2Lip

    model = Wav2Lip()
    ms_params = model.get_parameters()
    with open("pickles/pt_weights_wav2lip.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_wav2lip.ckpt")


def convert_lipreading():
    from tests.test_lipreading import load_args, get_model_from_json

    ms2pt = {}
    with open("tools/lipreading_mapping.txt", "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            p_torch = line.split("\t")[0]
            p_ms = line.split("\t")[1]
            ms2pt[p_ms] = p_torch
    ms2pt[".bn2d"] = ""

    args = load_args()
    model = get_model_from_json(args)
    ms_params = model.get_parameters()

    with open("pickles/lrw_resnet18_dctcn_audio.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/lipreading/ms_resnet18_dctcn_audio.ckpt", ms2pt)


def convert_hopenet():
    from models.face3d.facexlib.resnet import Bottleneck
    from models.facerender.networks import Hopenet

    hopenet = Hopenet(Bottleneck, [3, 4, 6, 3], 66)
    ms_params = hopenet.get_parameters()

    with open("./pt_weights_hopenet.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict,
                  "checkpoints/ms/ms_hopenet_robust_alpha1.ckpt")


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="CPU", device_id=2)
    convert_hopenet()
