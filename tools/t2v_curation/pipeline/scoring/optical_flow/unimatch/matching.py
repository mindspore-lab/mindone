import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .geometry import coords_grid, generate_window_grid, normalize_coords

def global_correlation_softmax(
    feature0,
    feature1,
    pred_bidir_flow=False,
):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.reshape(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.reshape(b, c, -1)  # [B, C, H*W]

    correlation = ops.matmul(feature0, feature1).reshape(b, h, w, h, w) / (c**0.5)  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(b, h, w)  # [B, 2, H, W]
    grid = init_grid.reshape(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.reshape(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = ops.cat((correlation, correlation.permute(0, 2, 1)), axis=0)  # [2*B, H*W, H*W]
        init_grid = init_grid.tile((2, 1, 1, 1))  # [2*B, 2, H, W]
        grid = grid.tile((2, 1, 1))  # [2*B, H*W, 2]
        b = b * 2


    prob = ops.softmax(correlation, axis=-1)  # [B, H*W, H*W]

    correspondence = ops.matmul(prob, grid).reshape(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob


def local_correlation_softmax(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
):
    b, c, h, w = feature0.shape
    coords_init = coords_grid(b, h, w)  # [B, 2, H, W]
    coords = coords_init.reshape(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(
        -local_radius, local_radius, -local_radius, local_radius, local_h, local_w
    )  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).tile((b, 1, 1, 1))  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = ms.Tensor(sample_coords[:, :, :, 0] >= 0, dtype=ms.int8) & ms.Tensor(sample_coords[:, :, :, 0] < w, dtype=ms.int8)  # [B, H*W, (2R+1)^2]
    valid_y = ms.Tensor(sample_coords[:, :, :, 1] >= 0, dtype=ms.int8) & ms.Tensor(sample_coords[:, :, :, 1] < h, dtype=ms.int8)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = ops.grid_sample(feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True).permute(
        0, 2, 1, 3
    )  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).reshape(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = ops.matmul(feature0_view, window_feature).reshape(b, h * w, -1) / (c**0.5)  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = ops.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = (
        ops.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).reshape(b, h, w, 2).permute(0, 3, 1, 2)
    )  # [B, 2, H, W]

    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob


def local_correlation_with_flow(
    feature0,
    feature1,
    flow,
    local_radius,
    padding_mode="zeros",
    dilation=1,
):
    b, c, h, w = feature0.shape
    coords_init = coords_grid(b, h, w)  # [B, 2, H, W]
    coords = coords_init.reshape(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(
        -local_radius, local_radius, -local_radius, local_radius, local_h, local_w
    )  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).tile((b, 1, 1, 1))  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid * dilation  # [B, H*W, (2R+1)^2, 2]

    # flow can be zero when using features after transformer
    if not isinstance(flow, float):
        sample_coords = sample_coords + flow.reshape(b, 2, -1).permute(0, 2, 1).unsqueeze(-2)  # [B, H*W, (2R+1)^2, 2]
    else:
        assert flow == 0.0

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = ops.grid_sample(feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True).permute(
        0, 2, 1, 3
    )  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).reshape(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = ops.matmul(feature0_view, window_feature).reshape(b, h * w, -1) / (c**0.5)  # [B, H*W, (2R+1)^2]

    corr = corr.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # [B, (2R+1)^2, H, W]

    return corr


def global_correlation_softmax_stereo(
    feature0,
    feature1,
):
    # global correlation on horizontal direction
    b, c, h, w = feature0.shape

    x_grid = ops.linspace(0, w - 1, w)  # [W]

    feature0 = feature0.permute(0, 2, 3, 1)  # [B, H, W, C]
    feature1 = feature1.permute(0, 2, 1, 3)  # [B, H, C, W]

    correlation = ops.matmul(feature0, feature1) / (c**0.5)  # [B, H, W, W]

    # mask subsequent positions to make disparity positive
    mask = ops.triu(ops.ones((w, w)), diagonal=1).astype(feature0.dtype)  # [W, W]
    valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).tile((b, h, 1, 1))  # [B, H, W, W]

    correlation[~valid_mask] = -1e9

    prob = ops.softmax(correlation, axis=-1)  # [B, H, W, W]

    correspondence = (x_grid.reshape(1, 1, 1, w) * prob).sum(-1)  # [B, H, W]

    # NOTE: unlike flow, disparity is typically positive
    disparity = x_grid.reshape(1, 1, w).tile((b, h, 1)) - correspondence  # [B, H, W]

    return disparity.unsqueeze(1), prob  # feature resolution

def local_correlation_softmax_stereo(
    feature0,
    feature1,
    local_radius,
):
    b, c, h, w = feature0.shape
    coords_init = coords_grid(b, h, w)  # [B, 2, H, W]
    coords = coords_init.reshape(b, 2, -1).permute(0, 2, 1).contiguous()  # [B, H*W, 2]

    local_h = 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(
        0, 0, -local_radius, local_radius, local_h, local_w
    )  # [1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).tile((b, 1, 1, 1))  # [B, 1, (2R+1), 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1), 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = ops.grid_sample(feature1, sample_coords_norm, padding_mode="zeros", align_corners=True).permute(
        0, 2, 1, 3
    )  # [B, H*W, C, (2R+1)]
    feature0_view = feature0.permute(0, 2, 3, 1).contiguous().reshape(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = ops.matmul(feature0_view, window_feature).reshape(b, h * w, -1) / (c**0.5)  # [B, H*W, (2R+1)]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = ops.softmax(corr, axis = -1)  # [B, H*W, (2R+1)]

    correspondence = (
        ops.matmul(prob.unsqueeze(-2), sample_coords_softmax)
        .squeeze(-2)
        .reshape(b, h, w, 2)
        .permute(0, 3, 1, 2)
        .contiguous()
    )  # [B, 2, H, W]

    flow = correspondence - coords_init  # flow at feature resolution
    match_prob = prob

    flow_x = -flow[:, :1]  # [B, 1, H, W]

    return flow_x, match_prob


def correlation_softmax_depth(
    feature0,
    feature1,
    intrinsics,
    pose,
    depth_candidates,
    depth_from_argmax=False,
    pred_bidir_depth=False,
):
    b, c, h, w = feature0.shape
    assert depth_candidates.ndim == 4  # [B, D, H, W]
    scale_factor = c**0.5

    if pred_bidir_depth:
        feature0, feature1 = ops.cat((feature0, feature1), axis=0), ops.cat((feature1, feature0), axis=0)
        intrinsics = intrinsics.tile((2, 1, 1))
        pose = ops.cat((pose, ops.inverse(pose)), axis=0)
        depth_candidates = depth_candidates.tile((2, 1, 1, 1))

    # depth candidates are actually inverse depth
    warped_feature1 = warp_with_pose_depth_candidates(
        feature1,
        intrinsics,
        pose,
        1.0 / depth_candidates,
    )  # [B, C, D, H, W]

    correlation = (feature0.unsqueeze(2) * warped_feature1).sum(1) / scale_factor  # [B, D, H, W]

    match_prob = ops.softmax(correlation, axis=1)  # [B, D, H, W]

    # for cross-task transfer (flow -> depth), extract depth with argmax at test time
    if depth_from_argmax:
        index = ops.argmax(match_prob, dim=1, keepdim=True)
        depth = ops.gather(depth_candidates, index=index, axis=1)
    else:
        depth = (match_prob * depth_candidates).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    return depth, match_prob


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.shape[1] == intrinsics.shape[2] == 3
    assert pose.shape[1] == pose.shape[2] == 4
    assert depth.ndim == 4

    b, d, h, w = depth.shape
    c = feature1.shape[1]

    # pixel coordinates
    grid = coords_grid(b, h, w, homogeneous=True)  # [B, 3, H, W]
    # back project to 3D and transform viewpoint
    points = ops.inverse(intrinsics).bmm(grid.reshape(b, 3, -1))  # [B, 3, H*W]
    points = ops.bmm(pose[:, :3, :3], points).unsqueeze(2).tile((1, 1, d, 1)) * depth.reshape(
        b, 1, d, h * w
    )  # [B, 3, D, H*W]
    points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
    # reproject to 2D image plane
    points = ops.bmm(intrinsics, points.reshape(b, 3, -1)).reshape(b, 3, d, h * w)  # [B, 3, D, H*W]
    pixel_coords = points[:, :2] / points[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]

    # normalize to [-1, 1]
    x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

    grid = ops.stack([x_grid, y_grid], axis=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = ops.grid_sample(
        feature1, grid.reshape(b, d * h, w, 2), mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature
