import os
import cv2
import yaml
import imageio
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from pydub import AudioSegment
from skimage import img_as_ubyte

from models.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from models.facerender.modules.mapping import MappingNet
from models.facerender.modules.generator import OcclusionAwareSPADEGenerator
from models.facerender.modules.make_animation import (
    make_animation,
)

from utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from utils.paste_pic import paste_pic
from utils.videoio import save_video_with_watermark


def load_cpk_facevid2vid(config, generator=None, kp_detector=None, he_estimator=None):
    ckpt_dir = config.path.get("checkpoint_dir", "")
    generator_path = os.path.join(ckpt_dir, config.path.get("generator_checkpoint", None))
    kp_extractor_path = os.path.join(ckpt_dir, config.path.get("kp_extractor_checkpoint", None))
    he_estimator_path = os.path.join(ckpt_dir, config.path.get("he_estimator_checkpoint", None))

    if (generator_path is not None) and (generator is not None):
        generator_params = ms.load_checkpoint(generator_path)
        ms.load_param_into_net(generator, generator_params)
        print(f"Finished loading the pretrained checkpoint {generator_path} into Generator.")
    if (kp_extractor_path is not None) and (kp_detector is not None):
        detector_params = ms.load_checkpoint(kp_extractor_path)
        ms.load_param_into_net(kp_detector, detector_params)
        print(f"Finished loading the pretrained checkpoint {kp_extractor_path} into KPDetector.")
    if (he_estimator_path is not None) and (he_estimator is not None):
        estimator_params = ms.load_checkpoint(he_estimator_path)
        ms.load_param_into_net(he_estimator, estimator_params)
        print(f"Finished loading the pretrained checkpoint {he_estimator_path} into HEEstimator.")


def load_cpk_mapping(config, mapping):
    ckpt_dir = config.path.get("checkpoint_dir", "")
    checkpoint_path = os.path.join(ckpt_dir, config.path.get("mappingnet_checkpoint", None))
    mapping_params = ms.load_checkpoint(checkpoint_path)
    ms.load_param_into_net(mapping, mapping_params)


class AnimateFromCoeff:
    def __init__(self, config):
        generator = OcclusionAwareSPADEGenerator(
            **config.model_params.generator_params, **config.model_params.common_params
        )
        kp_extractor = KPDetector(
            **config.model_params.kp_detector_params,
            **config.model_params.common_params,
        )
        he_estimator = HEEstimator(
            **config.model_params.he_estimator_params,
            **config.model_params.common_params,
        )
        mapping = MappingNet(**config.model_params.mapping_params)

        for param in generator.get_parameters():
            param.requires_grad = False
        for param in kp_extractor.get_parameters():
            param.requires_grad = False
        for param in he_estimator.get_parameters():
            param.requires_grad = False
        for param in mapping.get_parameters():
            param.requires_grad = False

        load_cpk_facevid2vid(config, generator, kp_extractor, he_estimator)
        load_cpk_mapping(config, mapping=mapping)

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.he_estimator = he_estimator
        self.mapping = mapping

        self.kp_extractor.set_train(False)
        self.generator.set_train(False)
        self.he_estimator.set_train(False)
        self.mapping.set_train(False)

    def generate(
        self,
        x,
        video_save_dir,
        pic_path,
        crop_info,
        enhancer=None,
        background_enhancer=None,
        preprocess="crop",
        img_size=256,
    ):
        source_image = x["source_image"]
        source_semantics = x["source_semantics"]
        target_semantics = x["target_semantics"]

        if "yaw_c_seq" in x:
            yaw_c_seq = x["yaw_c_seq"]
        else:
            yaw_c_seq = None
        if "pitch_c_seq" in x:
            pitch_c_seq = x["pitch_c_seq"]
        else:
            pitch_c_seq = None
        if "roll_c_seq" in x:
            roll_c_seq = x["roll_c_seq"]
        else:
            roll_c_seq = None

        frame_num = x["frame_num"]

        predictions_video = make_animation(
            source_image,
            source_semantics,
            target_semantics,
            self.generator,
            self.kp_extractor,
            self.he_estimator,
            self.mapping,
            yaw_c_seq,
            pitch_c_seq,
            roll_c_seq,
            use_exp=True,
        )

        predictions_video = predictions_video.reshape((-1,) + predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]

        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            image = np.transpose(image.asnumpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)

        result = img_as_ubyte(video)

        # the generated video is 256x256, so we keep the aspect ratio,
        original_size = crop_info[0]
        if original_size:
            result = [
                cv2.resize(
                    result_i,
                    (img_size, int(img_size * original_size[1] / original_size[0])),
                )
                for result_i in result
            ]

        video_name = x["video_name"] + ".mp4"
        path = os.path.join(video_save_dir, "temp_" + video_name)

        imageio.mimsave(path, result, fps=float(25))

        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path

        audio_path = x["audio_path"]
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name + ".wav")
        start_time = 0
        # cog will not keep the .mp3 filename
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num
        end_time = start_time + frames * 1 / 25 * 1000
        word1 = sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")

        save_video_with_watermark(path, new_audio_path, av_path, watermark=False)
        print(f"The generated video is named {video_save_dir}/{video_name}")

        if "full" in preprocess.lower():
            # only add watermark to the full image.
            video_name_full = x["video_name"] + "_full.mp4"
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(
                path,
                pic_path,
                crop_info,
                new_audio_path,
                full_video_path,
                extended_crop=True if "ext" in preprocess.lower() else False,
            )
            print(f"The generated video is named {video_save_dir}/{video_name_full}")
        else:
            full_video_path = av_path

        # paste back then enhancers
        if enhancer:
            video_name_enhancer = x["video_name"] + "_enhanced.mp4"
            enhanced_path = os.path.join(video_save_dir, "temp_" + video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer)
            return_path = av_path_enhancer

            try:
                enhanced_images_gen_with_len = enhancer_generator_with_len(
                    full_video_path, method=enhancer, bg_upsampler=background_enhancer
                )
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))

            except:
                enhanced_images_gen_with_len = enhancer_list(
                    full_video_path, method=enhancer, bg_upsampler=background_enhancer
                )
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))

            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark=False)
            print(f"The generated video is named {video_save_dir}/{video_name_enhancer}")
            os.remove(enhanced_path)

        os.remove(path)
        os.remove(new_audio_path)

        return return_path


class AnimateModel(nn.Cell):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, config):
        super(AnimateModel, self).__init__()

        generator = OcclusionAwareSPADEGenerator(
            **config.model_params.generator_params, **config.model_params.common_params
        )
        kp_extractor = KPDetector(
            **config.model_params.kp_detector_params,
            **config.model_params.common_params,
        )
        mapping = MappingNet(**config.model_params.mapping_params)

        load_cpk_facevid2vid(config, generator, kp_extractor)
        load_cpk_mapping(config, mapping=mapping)

        self.zeros_mat = ops.zeros((2, 1), dtype=ms.float32)
        self.ones_mat = ops.ones((2, 1), dtype=ms.float32)

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.set_train(True)
        self.generator.set_train(True)
        self.mapping.set_train(True)

    def keypoint_transformation_train(self, kp_value, he):
        yaw, pitch, roll, t, exp = he

        yaw = self.headpose_pred_to_degree(yaw)
        pitch = self.headpose_pred_to_degree(pitch)
        roll = self.headpose_pred_to_degree(roll)

        rot_mat = self.get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

        # keypoint rotation
        # kp_rotated = ms.Tensor(np.einsum("bmp,bkp->bkm", rot_mat.asnumpy(), kp.asnumpy()))
        rot_mat = ops.Cast()(rot_mat, ms.float32)
        kp_rotated = ops.BatchMatMul(transpose_b=True)(rot_mat, kp_value).transpose(0, 2, 1)

        # keypoint translation
        t[:, 0] = t[:, 0] * 0.0
        t[:, 2] = t[:, 2] * 0.0
        t = t.unsqueeze(1).repeat(kp_value.shape[1], axis=1)
        kp_t = kp_rotated + t

        # add expression deviation
        exp = exp.view(exp.shape[0], -1, 3)
        kp_transformed = kp_t + exp

        return kp_transformed

    def headpose_pred_to_degree(self, pred):
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = ms.Tensor(idx_tensor, dtype=ms.float32)
        pred = ops.softmax(pred)
        degree = ops.sum(pred * idx_tensor, 1) * 3.0 - 99.0
        return degree

    def get_rotation_matrix(self, yaw, pitch, roll):
        yaw = yaw / 180.0 * 3.14
        pitch = pitch / 180.0 * 3.14
        roll = roll / 180.0 * 3.14

        roll = roll.unsqueeze(1).astype(ms.float32)
        pitch = pitch.unsqueeze(1).astype(ms.float32)
        yaw = yaw.unsqueeze(1).astype(ms.float32)

        pitch_mat = ops.cat(
            [
                self.ones_mat,
                self.zeros_mat,
                self.zeros_mat,
                self.zeros_mat,
                ops.cos(pitch),
                -ops.sin(pitch),
                self.zeros_mat,
                ops.sin(pitch),
                ops.cos(pitch),
            ],
            axis=1,
        )

        pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

        yaw_mat = ops.cat(
            [
                ops.cos(yaw),
                self.zeros_mat,
                ops.sin(yaw),
                self.zeros_mat,
                self.ones_mat,
                self.zeros_mat,
                -ops.sin(yaw),
                self.zeros_mat,
                ops.cos(yaw),
            ],
            axis=1,
        )

        yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

        roll_mat = ops.cat(
            [
                ops.cos(roll),
                -ops.sin(roll),
                self.zeros_mat,
                ops.sin(roll),
                ops.cos(roll),
                self.zeros_mat,
                self.zeros_mat,
                self.zeros_mat,
                self.ones_mat,
            ],
            axis=1,
        )

        roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

        mid_mat = ops.BatchMatMul(transpose_b=True)(pitch_mat, yaw_mat)
        rot_mat = ops.BatchMatMul(transpose_b=True)(mid_mat, roll_mat)

        return rot_mat

    def construct(self, source_image, source_semantics, target_semantics):
        kp_canonical = self.kp_extractor(source_image)  # value
        he_source = self.mapping(source_semantics)  # (yaw, pitch, roll, t, exp)
        kp_source = self.keypoint_transformation_train(kp_canonical, he_source)
        he_driving = self.mapping(target_semantics)
        kp_driving = self.keypoint_transformation_train(kp_canonical, he_driving)

        out = self.generator(source_image, kp_source=kp_source, kp_driving=kp_driving)

        return out, kp_canonical, kp_source, kp_driving, he_source, he_driving
