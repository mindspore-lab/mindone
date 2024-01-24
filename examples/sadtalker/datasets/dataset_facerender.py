import os
import random
import numpy as np
from PIL import Image
from skimage import io, img_as_float32, transform
import mindspore as ms
from mindspore import ops
import scipy.io as scio
from glob import glob

from models.facerender.modules.utils import make_coordinate_grid_2d
from utils.get_file import get_img_paths


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


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """

    def __init__(self, bs, **kwargs):
        noise = ops.normal(shape=(2, 3), mean=0, stddev=kwargs["sigma_affine"])
        self.theta = noise + ops.eye(2, 3)
        self.bs = 1

        if ("sigma_tps" in kwargs) and ("points_tps" in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d(
                (kwargs["points_tps"], kwargs["points_tps"]), type=noise.dtype
            )
            # self.control_points = self.control_points.unsqueeze(0)
            self.control_params = ops.normal(
                shape=(1, kwargs["points_tps"] ** 2),
                mean=0,
                stddev=kwargs["sigma_tps"],
            )
        else:
            self.tps = False

    def transform_frame(self, frame):
        datatype = frame.dtype
        grid = make_coordinate_grid_2d(frame.shape[1:], type=datatype).unsqueeze(0)
        grid = grid.view(frame.shape[1] * frame.shape[2], 2)
        grid = self.warp_coordinates(grid).view(frame.shape[1], frame.shape[2], 2)
        res_frame = ops.grid_sample(frame.unsqueeze(0), grid.unsqueeze(0), padding_mode="reflection").squeeze(0)
        return res_frame

    def warp_coordinates(self, coordinates):
        datatype = coordinates.dtype
        theta = self.theta.astype(datatype)
        theta = theta.unsqueeze(0)
        transformed = ops.matmul(theta[:, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.astype(datatype)
            control_params = self.control_params.astype(datatype)
            distances = coordinates.view(-1, 1, 2) - control_points.view(1, -1, 2)
            distances = ops.abs(distances).sum(-1)

            result = distances**2
            result = result * ops.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(axis=1).view(coordinates.shape[0], 1)
            transformed = transformed + result

        transformed = ops.cast(transformed, datatype)

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = ops.grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = ops.grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = ops.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], axis=-2)
        return jacobian


class TestFaceRenderDataset:
    def __init__(
        self,
        args,
        config,
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
        self.cfg = config
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

    def prepare_source_features(self, pic_path, first_coeff_path, is_train=False):
        """prepare the source image features
        Take the 1st image in the video if it is training.
        """
        img1 = Image.open(pic_path)
        img1 = np.array(img1)

        source_image = img_as_float32(img1)
        source_image = transform.resize(source_image, (self.size, self.size, 3))
        source_image = source_image.transpose((2, 0, 1))

        source_semantics_dict = scio.loadmat(first_coeff_path)

        if "full" not in self.preprocess.lower():
            source_semantics = source_semantics_dict["coeff_3dmm"][:1, :70]  # 1 70

        else:
            source_semantics = source_semantics_dict["coeff_3dmm"][:1, :73]  # 1 70

        source_semantics_new = transform_semantic_1(source_semantics, self.semantic_radius)
        source_semantics_new = np.asarray(source_semantics_new).astype("float32")

        if not is_train:
            source_image_ts = ms.Tensor(source_image).unsqueeze(0)
            source_image_ts = source_image_ts.repeat(self.batch_size, axis=0)
            source_semantics_ts = ms.Tensor(source_semantics_new).unsqueeze(0)
            source_semantics_ts = source_semantics_ts.repeat(self.batch_size, axis=0)
        else:
            source_image_ts = ms.Tensor(source_image)
            source_semantics_ts = ms.Tensor(source_semantics_new)

        return source_image_ts, source_semantics, source_semantics_ts, img1

    def prepare_target_features(self, coeff_path, source_semantics, frame_idx=None, tgt_img_path=None):
        """prepare the driving audio features"""
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

        if frame_idx is None:  # test
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

        else:  # train
            target_semantics = transform_semantic_target(generated_3dmm, frame_idx, self.semantic_radius)
            target_semantics_np = np.array(target_semantics)
            target_semantics_np = np.asarray(target_semantics_np).astype("float32")
            target_semantics_ts = ms.Tensor(target_semantics_np)

        if tgt_img_path is not None:  # train
            tgt_img = Image.open(tgt_img_path)
            target_image = np.array(tgt_img)
            target_image = img_as_float32(target_image)
            target_image = transform.resize(target_image, (self.size, self.size, 3))
            target_image = target_image.transpose((2, 0, 1))
            target_image_ts = ms.Tensor(target_image)

        else:  # test
            target_image_ts = None

        return frame_num, target_semantics_ts, target_image_ts

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
            source_image_binary,
        ) = self.prepare_source_features(self.pic_path, self.first_coeff_path)
        data["source_image"] = source_image_ts
        data["source_semantics"] = source_semantics_ts

        # target
        frame_num, target_semantics_ts, _ = self.prepare_target_features(self.coeff_path, source_semantics)

        video_name = os.path.splitext(os.path.split(self.coeff_path)[-1])[0]

        data["target_semantics"] = target_semantics_ts
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
        config,
        train_list,  # (img_folder, first_coeff_path, net_coeff_path)
        batch_size,
        expression_scale=1.0,
        still_mode=False,
        preprocess="crop",
        size=256,
        semantic_radius=13,
        syncnet_T=5,
        extractor=None,
    ):
        super().__init__(
            args,
            config,
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

        self.extractor = extractor
        self.all_videos = get_img_paths(train_list)
        self.syncnet_T = syncnet_T
        self.output_columns = [
            "source_image",
            "source_semantics",
            "transformed_src_semantics",
            "target_semantics",
            "target_image_ts",
            "frame_num",
        ]

        # transform for equivariance loss
        self.equi_transform = Transform(batch_size, **config.facerender.train.transform_params)

    def get_output_columns(self):
        return self.output_columns

    def transform_equivariance(self, image_binary):
        len_coeff = 70 if "full" not in self.preprocess.lower() else 73

        transformed_frame = self.equi_transform.transform_frame(
            ms.Tensor(image_binary, ms.float32)
        )  # (bs, 256, 256, 3)
        coeff_dict = self.extractor.extract_3dmm([transformed_frame])
        transformed_semantics = coeff_dict["coeff_3dmm"][:, :len_coeff]
        transformed_semantics = ms.Tensor(transformed_semantics, ms.float32).unsqueeze(-1)
        transformed_semantics = transformed_semantics.repeat(self.semantic_radius * 2 + 1, axis=-1)
        return transformed_semantics.squeeze(0)

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
        src_img_path = valid_paths[0]
        tgt_img_path = random.choice(valid_paths)  # 随机选取一帧
        frame_id = int(os.path.basename(tgt_img_path).split("_")[-1].split(".")[0])

        # source
        (
            source_image_ts,
            source_semantics,
            source_semantics_ts,
            source_image_binary,
        ) = self.prepare_source_features(src_img_path, first_coeff_path, is_train=True)

        data["source_image"] = source_image_ts
        data["source_semantics"] = source_semantics_ts

        # transform
        transformed_src_semantics = self.transform_equivariance(source_image_binary)
        data["transformed_src_semantics"] = transformed_src_semantics

        # target
        frame_num, target_semantics_ts, target_image_ts = self.prepare_target_features(
            net_coeff_path, source_semantics, frame_id, tgt_img_path
        )

        data["target_semantics"] = target_semantics_ts
        data["target_image_ts"] = target_image_ts
        data["frame_num"] = frame_num

        input_yaw_list = self.args.input_yaw
        input_pitch_list = self.args.input_pitch
        input_roll_list = self.args.input_roll

        if input_yaw_list is not None:
            yaw_c_seq = gen_camera_pose(input_yaw_list, frame_num, self.batch_size)
            data["yaw_c_seq"] = ms.Tensor(yaw_c_seq)
            self.output_columns.append("yaw_c_seq")
        if input_pitch_list is not None:
            pitch_c_seq = gen_camera_pose(input_pitch_list, frame_num, self.batch_size)
            data["pitch_c_seq"] = ms.Tensor(pitch_c_seq)
            self.output_columns.append("pitch_c_seq")
        if input_roll_list is not None:
            roll_c_seq = gen_camera_pose(input_roll_list, frame_num, self.batch_size)
            data["roll_c_seq"] = ms.Tensor(roll_c_seq)
            self.output_columns.append("roll_c_seq")

        output_tuple = tuple(data[k] for k in self.output_columns)

        return output_tuple
