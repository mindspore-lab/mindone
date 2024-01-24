from scipy.spatial import ConvexHull
import mindspore as ms
from mindspore import nn, ops
import numpy as np
from tqdm import tqdm


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale=False,
    use_relative_movement=False,
    use_relative_jacobian=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source[0][0].asnumpy()).volume
        driving_area = ConvexHull(kp_driving_initial[0][0].asnumpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = kp_driving.copy()

    if use_relative_movement:
        kp_value_diff = kp_driving[0] - kp_driving_initial[0]
        kp_value_diff *= adapt_movement_scale
        kp_new[0] = kp_value_diff + kp_source[0]

        if use_relative_jacobian:
            jacobian_diff = ops.MatMul()(kp_driving[1], ops.inverse(kp_driving_initial[1]))
            kp_new[1] = ops.MatMul()(jacobian_diff, kp_source[1])

    return kp_new


def headpose_pred_to_degree(pred):
    datatype = pred.dtype
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = ms.Tensor(idx_tensor, dtype=datatype)
    pred = ops.softmax(pred)
    degree = ops.sum(pred * idx_tensor, 1) * 3.0 - 99.0
    degree = ops.cast(degree, datatype)
    return degree


def get_rotation_matrix(yaw, pitch, roll, datatype=ms.float32):
    yaw = yaw / 180.0 * 3.14
    pitch = pitch / 180.0 * 3.14
    roll = roll / 180.0 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = ops.cat(
        [
            ops.ones((2, 1), dtype=ms.float32),
            ops.zeros((2, 1), dtype=ms.float32),
            ops.zeros((2, 1), dtype=ms.float32),
            ops.zeros((2, 1), dtype=ms.float32),
            ops.cos(pitch),
            -ops.cos(pitch),
            ops.zeros((2, 1), dtype=ms.float32),
            ops.sin(pitch),
            ops.cos(pitch),
        ],
        axis=1,
    )

    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = ops.cat(
        [
            ops.Cos()(yaw),
            ops.zeros((2, 1), dtype=yaw.dtype),
            ops.Sin()(yaw),
            ops.zeros((2, 1), dtype=yaw.dtype),
            ops.ones((2, 1), dtype=yaw.dtype),
            ops.zeros((2, 1), dtype=yaw.dtype),
            -ops.Sin()(yaw),
            ops.zeros((2, 1), dtype=yaw.dtype),
            ops.Cos()(yaw),
        ],
        axis=1,
    )

    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = ops.cat(
        [
            ops.Cos()(roll),
            -ops.Sin()(roll),
            ops.zeros((2, 1), dtype=roll.dtype),
            ops.Sin()(roll),
            ops.Cos()(roll),
            ops.zeros((2, 1), dtype=roll.dtype),
            ops.zeros((2, 1), dtype=roll.dtype),
            ops.zeros((2, 1), dtype=roll.dtype),
            ops.ones((2, 1), dtype=roll.dtype),
        ],
        axis=1,
    )

    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    # rot_mat = ms.Tensor(
    #     np.einsum(
    #         "bij,bjk,bkm->bim",
    #         pitch_mat.asnumpy(),
    #         yaw_mat.asnumpy(),
    #         roll_mat.asnumpy(),
    #     )
    # )

    mid_mat = ops.BatchMatMul(transpose_b=True)(pitch_mat, yaw_mat)
    rot_mat = ops.BatchMatMul(transpose_b=True)(mid_mat, roll_mat)

    return rot_mat


def keypoint_transformation(kp_canonical, he, wo_exp=False, yaw_in=None, pitch_in=None, roll_in=None):
    kp_value = kp_canonical.astype(ms.float32)  # (bs, k, 3)
    yaw, pitch, roll, t, exp = he

    # yaw, pitch, roll = he["yaw"], he["pitch"], he["roll"]
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if yaw_in is not None:
        yaw = yaw_in
    if pitch_in is not None:
        pitch = pitch_in
    if roll_in is not None:
        roll = roll_in

    rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

    if wo_exp:
        exp = exp * 0.0

    # keypoint rotation
    # kp_rotated = ms.Tensor(np.einsum("bmp,bkp->bkm", rot_mat.asnumpy(), kp.asnumpy()))
    rot_mat = rot_mat.astype(ms.float32)
    kp_rotated = ops.BatchMatMul(transpose_b=True)(rot_mat, kp_value).transpose(0, 2, 1).astype(ms.float32)

    # keypoint translation
    t[:, 0] = t[:, 0] * 0.0
    t[:, 2] = t[:, 2] * 0.0
    t = t.unsqueeze(1).repeat(kp_value.shape[1], axis=1)
    kp_t = kp_rotated + t

    # add expression deviation
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return kp_transformed


def make_animation(
    source_image,
    source_semantics,
    target_semantics,
    generator,
    kp_detector,
    he_estimator,
    mapping,
    yaw_c_seq=None,
    pitch_c_seq=None,
    roll_c_seq=None,
    use_exp=True,
    use_half=False,
):
    predictions = []

    kp_canonical = kp_detector(source_image)
    he_source = mapping(source_semantics)
    kp_source = keypoint_transformation(kp_canonical, he_source)

    for frame_idx in tqdm(range(target_semantics.shape[1]), "Face Renderer:"):
        # still check the dimension
        # print(target_semantics.shape, source_semantics.shape)
        target_semantics_frame = target_semantics[:, frame_idx]
        he_driving = mapping(target_semantics_frame)

        yaw_in = None
        pitch_in = None
        roll_in = None

        if yaw_c_seq is not None:
            yaw_in = yaw_c_seq[:, frame_idx]
        if pitch_c_seq is not None:
            pitch_in = pitch_c_seq[:, frame_idx]
        if roll_c_seq is not None:
            roll_in = roll_c_seq[:, frame_idx]

        kp_driving = keypoint_transformation(
            kp_canonical, he_driving, yaw_in=yaw_in, pitch_in=pitch_in, roll_in=roll_in
        )

        out = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)

        """
        source_image_new = out['prediction'].squeeze(1)
        kp_canonical_new =  kp_detector(source_image_new)
        he_source_new = he_estimator(source_image_new)
        kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
        kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
        out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
        """
        predictions.append(out)
    predictions_ts = ops.stack(predictions, axis=1)
    return predictions_ts
