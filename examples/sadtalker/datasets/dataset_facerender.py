import os
import random
import numpy as np
from PIL import Image
from skimage import io, img_as_float32, transform
import mindspore as ms
import scipy.io as scio
from utils.get_file import get_img_paths
from glob import glob


def transform_semantic_1(semantic, semantic_radius):
    semantic_list = [semantic for i in range(0, semantic_radius * 2 + 1)]
    coeff_3dmm = np.concatenate(semantic_list, 0)
    return coeff_3dmm.transpose(1, 0)


def transform_semantic_target(coeff_3dmm, frame_index, semantic_radius):
    num_frames = coeff_3dmm.shape[0]
    seq = list(range(frame_index - semantic_radius, frame_index + semantic_radius + 1))
    index = [min(max(item, 0), num_frames - 1) for item in seq]
    coeff_3dmm_g = coeff_3dmm[index, :]
    return coeff_3dmm_g.transpose(1, 0)


def gen_camera_pose(camera_degree_list, frame_num, batch_size):
    new_degree_list = []
    if len(camera_degree_list) == 1:
        for _ in range(frame_num):
            new_degree_list.append(camera_degree_list[0])
        remainder = frame_num % batch_size
        if remainder != 0:
            for _ in range(batch_size - remainder):
                new_degree_list.append(new_degree_list[-1])
        new_degree_np = np.array(new_degree_list).reshape(batch_size, -1)
        return new_degree_np

    degree_sum = 0.0
    for i, degree in enumerate(camera_degree_list[1:]):
        degree_sum += abs(degree - camera_degree_list[i])

    degree_per_frame = degree_sum / (frame_num - 1)
    for i, degree in enumerate(camera_degree_list[1:]):
        degree_last = camera_degree_list[i]
        degree_step = degree_per_frame * abs(degree - degree_last) / (degree - degree_last)
        new_degree_list = new_degree_list + list(np.arange(degree_last, degree, degree_step))
    if len(new_degree_list) > frame_num:
        new_degree_list = new_degree_list[:frame_num]
    elif len(new_degree_list) < frame_num:
        for _ in range(frame_num - len(new_degree_list)):
            new_degree_list.append(new_degree_list[-1])
    print(len(new_degree_list))
    print(frame_num)

    remainder = frame_num % batch_size
    if remainder != 0:
        for _ in range(batch_size - remainder):
            new_degree_list.append(new_degree_list[-1])
    new_degree_np = np.array(new_degree_list).reshape(batch_size, -1)
    return new_degree_np


class TestFaceRenderDataset:
    def __init__(
        self,
        args,
        coeff_path,
        pic_path,
        first_coeff_path,
        audio_path,
        batch_size,
        expression_scale=1.0,
        still_mode=False,
        preprocess="crop",
        size=256,
        semantic_radius=13,
    ) -> None:
        self.args = args
        self.coeff_path = coeff_path
        self.pic_path = pic_path
        self.first_coeff_path = first_coeff_path
        self.audio_path = audio_path
        self.batch_size = batch_size
        self.expression_scale = expression_scale
        self.still_mode = still_mode
        self.preprocess = preprocess
        self.size = size
        self.semantic_radius = semantic_radius

    def prepare_source_features(self, pic_path, first_coeff_path):
        img1 = Image.open(pic_path)
        source_image = np.array(img1)
        source_image = img_as_float32(source_image)
        source_image = transform.resize(source_image, (self.size, self.size, 3))
        source_image = source_image.transpose((2, 0, 1))
        source_image_ts = ms.Tensor(source_image).unsqueeze(0)
        source_image_ts = source_image_ts.repeat(self.batch_size, axis=0)

        source_semantics_dict = scio.loadmat(first_coeff_path)

        if "full" not in self.preprocess.lower():
            source_semantics = source_semantics_dict["coeff_3dmm"][:1, :70]  # 1 70

        else:
            source_semantics = source_semantics_dict["coeff_3dmm"][:1, :73]  # 1 70

        source_semantics_new = transform_semantic_1(source_semantics, self.semantic_radius)
        source_semantics_new = np.asarray(source_semantics_new).astype("float32")
        source_semantics_ts = ms.Tensor(source_semantics_new).unsqueeze(0)
        source_semantics_ts = source_semantics_ts.repeat(self.batch_size, axis=0)

        return source_image_ts, source_semantics, source_semantics_ts

    def prepare_target_features(self, coeff_path, source_semantics):
        txt_path = os.path.splitext(coeff_path)[0]

        generated_dict = scio.loadmat(coeff_path)
        generated_3dmm = generated_dict["coeff_3dmm"][:, :70]
        generated_3dmm[:, :64] = generated_3dmm[:, :64] * self.expression_scale

        if "full" in self.preprocess.lower():
            generated_3dmm = np.concatenate(
                [
                    generated_3dmm,
                    np.repeat(source_semantics[:, 70:], generated_3dmm.shape[0], axis=0),
                ],
                axis=1,
            )

        if self.still_mode:
            generated_3dmm[:, 64:] = np.repeat(source_semantics[:, 64:], generated_3dmm.shape[0], axis=0)

        with open(txt_path + ".txt", "w") as f:
            for coeff in generated_3dmm:
                for i in coeff:
                    f.write(str(i)[:7] + "  " + "\t")
                f.write("\n")

        target_semantics_list = []
        frame_num = generated_3dmm.shape[0]

        for frame_idx in range(frame_num):
            target_semantics = transform_semantic_target(generated_3dmm, frame_idx, self.semantic_radius)
            target_semantics_list.append(target_semantics)

        remainder = frame_num % self.batch_size
        if remainder != 0:
            for _ in range(self.batch_size - remainder):
                target_semantics_list.append(target_semantics)

        # frame_num 70 semantic_radius*2+1
        target_semantics_np = np.array(target_semantics_list)
        target_semantics_np = target_semantics_np.reshape(
            self.batch_size,
            -1,
            target_semantics_np.shape[-2],
            target_semantics_np.shape[-1],
        )

        target_semantics_np = np.asarray(target_semantics_np).astype("float32")
        target_semantics_ts = ms.Tensor(target_semantics_np)

        return frame_num, target_semantics_ts

    def __next__(self):
        if self._index >= 1:
            raise StopIteration
        else:
            item = self.__getitem__(self._index)
            self._index += 1
            return item

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = {}

        # source
        (
            source_image_ts,
            source_semantics,
            source_semantics_ts,
        ) = self.prepare_source_features(self.pic_path, self.first_coeff_path)
        data["source_image"] = source_image_ts
        data["source_semantics"] = source_semantics_ts

        # target
        frame_num, target_semantics_ts = self.prepare_target_features(self.coeff_path, source_semantics)

        video_name = os.path.splitext(os.path.split(self.coeff_path)[-1])[0]

        data["target_semantics_list"] = target_semantics_ts
        data["frame_num"] = frame_num
        data["video_name"] = video_name
        data["audio_path"] = self.audio_path

        input_yaw_list = self.args.input_yaw
        input_pitch_list = self.args.input_pitch
        input_roll_list = self.args.input_roll

        if input_yaw_list is not None:
            yaw_c_seq = gen_camera_pose(input_yaw_list, frame_num, self.batch_size)
            data["yaw_c_seq"] = ms.Tensor(yaw_c_seq)
        if input_pitch_list is not None:
            pitch_c_seq = gen_camera_pose(input_pitch_list, frame_num, self.batch_size)
            data["pitch_c_seq"] = ms.Tensor(pitch_c_seq)
        if input_roll_list is not None:
            roll_c_seq = gen_camera_pose(input_roll_list, frame_num, self.batch_size)
            data["roll_c_seq"] = ms.Tensor(roll_c_seq)

        return data


class TrainFaceRenderDataset(TestFaceRenderDataset):
    def __init__(
        self,
        args,
        train_list,  # (img_folder, first_coeff_path, net_coeff_path)
        batch_size,
        expression_scale=1.0,
        still_mode=False,
        preprocess="crop",
        size=256,
        semantic_radius=13,
        syncnet_T=5,
    ):
        super().__init__(
            args,
            batch_size=batch_size,
            coeff_path=None,
            pic_path=None,
            first_coeff_path=None,
            audio_path=None,
            expression_scale=expression_scale,
            still_mode=still_mode,
            preprocess=preprocess,
            size=size,
            semantic_radius=semantic_radius,
        )

        self.all_videos = get_img_paths(train_list)
        self.syncnet_T = syncnet_T

    def get_label_features(self):
        pass

    def __next__(self):
        if self._index >= len(self.all_videos):
            raise StopIteration
        else:
            item = self.__getitem__(self._index)
            self._index += 1
            return item

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        data = {}

        img_folder, first_coeff_path, net_coeff_path = self.all_videos[idx].split(" ")
        image_paths = glob(os.path.join(img_folder, "*.png"))  # 图像的路径

        ##随机选取一帧,得到窗口并读取图片
        valid_paths = list(sorted(image_paths))[: len(image_paths) - self.syncnet_T]
        img_path = random.choice(valid_paths)  # 随机选取一帧

        # source
        (
            source_image_ts,
            source_semantics,
            source_semantics_ts,
        ) = self.prepare_source_features(img_path, first_coeff_path)

        data["source_image"] = source_image_ts
        data["source_semantics"] = source_semantics_ts

        # target
        frame_num, target_semantics_ts = self.prepare_target_features(net_coeff_path, source_semantics)

        data["target_semantics_list"] = target_semantics_ts
        data["frame_num"] = frame_num

        input_yaw_list = self.args.input_yaw
        input_pitch_list = self.args.input_pitch
        input_roll_list = self.args.input_roll

        if input_yaw_list is not None:
            yaw_c_seq = gen_camera_pose(input_yaw_list, frame_num, self.batch_size)
            data["yaw_c_seq"] = ms.Tensor(yaw_c_seq)
        if input_pitch_list is not None:
            pitch_c_seq = gen_camera_pose(input_pitch_list, frame_num, self.batch_size)
            data["pitch_c_seq"] = ms.Tensor(pitch_c_seq)
        if input_roll_list is not None:
            roll_c_seq = gen_camera_pose(input_roll_list, frame_num, self.batch_size)
            data["roll_c_seq"] = ms.Tensor(roll_c_seq)

        data["img_gt"] = img_path

        return data
