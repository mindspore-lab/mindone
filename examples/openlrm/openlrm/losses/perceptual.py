# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from mindspore import nn

from .lpips import LPIPS

__all__ = ["LPIPSLoss"]


class LPIPSLoss(nn.Cell):
    """
    Compute LPIPS loss between two images.
    """

    def __init__(self, prefech: bool = False):
        super().__init__()
        self.cached_models = {}
        if prefech:
            self.prefetch_models()

    def _get_model(self, model_name: str):
        if model_name not in self.cached_models:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                _model = LPIPS()

            self.cached_models[model_name] = _model
        return self.cached_models[model_name]

    def prefetch_models(self):
        _model_names = ["vgg"]  # eval:'alex'(not supported yet),  train: 'vgg'
        for model_name in _model_names:
            self._get_model(model_name)

    def construct(self, x, y, is_training: bool = True):
        """
        Assume images are 0-1 scaled and channel first.

        Args:
            x: [N, M, C, H, W]
            y: [N, M, C, H, W]
            is_training: whether to use VGG or AlexNet.

        Returns:
            Mean-reduced LPIPS loss across batch.
        """
        model_name = "vgg"  # if is_training else 'alex'
        loss_fn = self._get_model(model_name)
        N, M, C, H, W = x.shape
        x = x.reshape(N * M, C, H, W)
        y = y.reshape(N * M, C, H, W)
        image_loss = loss_fn(x, y, normalize=True).mean(axis=[1, 2, 3])
        batch_loss = image_loss.reshape(N, M).mean(axis=1)
        all_loss = batch_loss.mean()
        return all_loss
