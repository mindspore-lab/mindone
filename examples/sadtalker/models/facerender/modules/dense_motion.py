from mindspore import nn, ops
from models.facerender.modules.utils import Hourglass, make_coordinate_grid, kp2gaussian
# from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
from mindspore.nn import BatchNorm3d


class DenseMotionNetwork(nn.Cell):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(
            num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters,
                              num_kp + 1, kernel_size=7, pad_mode='pad', padding=3, has_bias=True)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1, has_bias=True)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(
                self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, pad_mode='pad', padding=3, has_bias=True)
        else:
            self.occlusion = None

        self.num_kp = num_kp

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid(
            (d, h, w), type=kp_source['value'].dtype)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - \
            kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)

        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            jacobian = ops.matmul(
                kp_source['jacobian'], ops.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            # jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            jacobian = jacobian.repeat(d, axis=2).repeat(
                h, axis=3).repeat(w, axis=4)
            coordinate_grid = ops.matmul(
                jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + \
            kp_source['value'].view(
                bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, axis=0)
        sparse_motions = ops.cat(
            [identity_grid, driving_to_source], axis=1)  # bs num_kp+1 d h w 3

        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(
            self.num_kp+1, axis=1)      # (bs, num_kp+1, 1, c, d, h, w)
        # (bs*(num_kp+1), c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)
        # (bs*(num_kp+1), d, h, w, 3) !!!!
        sparse_motions = sparse_motions.view(
            (bs * (self.num_kp+1), d, h, w, -1))
        sparse_deformed = ops.grid_sample(feature_repeat, sparse_motions)
        # (bs, num_kp+1, c, d, h, w)
        sparse_deformed = sparse_deformed.view(
            (bs, self.num_kp+1, -1, d, h, w))
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(
            kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(
            kp_source, spatial_size=spatial_size, kp_variance=0.01)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = ops.zeros(
            (heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]), heatmap.dtype)
        heatmap = ops.cat([zeros, heatmap], axis=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def construct(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = ops.relu(feature)

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(
            feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        heatmap = self.create_heatmap_representations(
            deformed_feature, kp_driving, kp_source)

        input_ = ops.cat([heatmap, deformed_feature], axis=2)
        input_ = input_.view(bs, -1, d, h, w)

        # input = deformed_feature.view(bs, -1, d, h, w)      # (bs, num_kp+1 * c, d, h, w)

        prediction = self.hourglass(input_)

        mask = self.mask(prediction)
        mask = ops.softmax(mask, axis=1)
        out_dict['mask'] = mask
        # (bs, num_kp+1, 1, d, h, w)
        mask = mask.unsqueeze(2)

        zeros_mask = ops.zeros_like(mask)
        mask = ops.where(mask < 1e-3, zeros_mask, mask)

        sparse_motion = sparse_motion.permute(
            0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        # (bs, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(axis=1)
        deformation = deformation.permute(
            0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = ops.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
