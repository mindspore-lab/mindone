import numpy as np
import mindspore as ms

from examples.stable_diffusion_v2.ldm.modules.fatezero.utils import get_word_inds


class SpatialBlender:
    """
    Return a blending mask using the cross attention produced by both source during the inversion and target prompt during editing.
    Called in make_controller
    """

    def get_mask(self, maps, alpha, use_pool, h=None, w=None):
        """
        ([1, 40, 2, 16, 16, 77]) * ([1, 1, 1, 1, 1, 77]) -> [2, 1, 16, 16]
        mask have dimension of [clip_length, dim, res, res]
        """
        k = 1
        if maps.dim() == 5: alpha = alpha[:, None, ...]
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = ms.ops.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = ms.ops.interpolate(maps, size=(h, w), mode="nearest")
        mask = mask / mask.max(-2, keepdims=True)[0].max(-1, keepdims=True)[0]
        mask = mask.gt(0.9)
        return mask

    def __call__(self, attention_store, step_in_store: int = None, target_h=None, target_w=None):
        maps = attention_store["input_blocks_cross"][2:4] + attention_store["output_blocks_cross"][:3]
        rearranged_maps = []
        for item in maps:
            if len(item.shape) == 4:
                item = item[None, ...]
            (p, c, heads, r, w) = item.shape
            res_h = int(np.sqrt(r))
            assert r == res_h * res_h, "the shape of attention map must be a squire"
            ms.ops.reshape(item, (p, c, heads, res_h, res_h, w))
            ms.ops.transpose(item, (0, 2, 1, 3, 4, 5))
            rearranged_maps.append(item)
        maps = ms.ops.cat(rearranged_maps, axis=1)
        masked_alpah_layers = self.alpha_layers
        mask = self.get_mask(maps, masked_alpah_layers, True, target_h, target_w)
        mask = mask.float()

        # mask is one: use generated information
        # mask is zero: use inverted information
        return mask

    def __init__(self, prompts, words):
        self.count = 0
        self.MAX_NUM_WORDS = 77

        alpha_layers = ms.ops.zeros((len(prompts), 1, 1, 1, 1, self.MAX_NUM_WORDS))
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word)
                alpha_layers[i, :, :, :, :, ind] = 1
        # self.alpha_layers.shape = torch.Size([2, 1, 1, 1, 1, 77]), 1 denotes the world to be replaced

        self.alpha_layers = alpha_layers
        print('the index mask of edited word in the prompt')
        print(self.alpha_layers[0][..., 0:(len(prompts[0].split(" ")) + 2)])
        print(self.alpha_layers[1][..., 0:(len(prompts[1].split(" ")) + 2)])
        self.counter = 0
        self.mask_list = []
