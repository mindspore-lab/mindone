# Adapted from https://github.com/VectorSpaceLab/OmniGen2/blob/main/omnigen2/models/transformers/repo.py
from einops import repeat

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from mindone.diffusers.models.embeddings import get_1d_rotary_pos_embed


class OmniGen2RotaryPosEmbed(nn.Cell):
    def __init__(
        self,
        theta: int,
        axes_dim: tuple[int, int, int],
        axes_lens: tuple[int, int, int] = (300, 512, 512),
        patch_size: int = 2,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    def get_freqs_cis(axes_dim: tuple[int, int, int], axes_lens: tuple[int, int, int], theta: int) -> list[Tensor]:
        freqs_cis = []
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=ms.float64)  # TODO: check precision
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, freqs_cis, ids: Tensor) -> Tensor:
        result = []
        for i in range(len(self.axes_dim)):
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs_cis[i].shape[-1]).to(ms.int64)
            result.append(
                ops.gather(freqs_cis[i], axis=0, input_indices=index)[..., 0, :]  # FIXME: mint doesn't support complex
            )
        return mint.cat(result, dim=-1)

    def construct(
        self, freqs_cis, attention_mask, l_effective_ref_img_len, l_effective_img_len, ref_img_sizes, img_sizes
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [
            cap_len + sum(ref_img_len) + img_len
            for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # Create position IDs
        position_ids = mint.zeros((batch_size, max_seq_len, 3), dtype=ms.int32)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            # add text position ids
            position_ids[i, :cap_seq_len] = repeat(mint.arange(cap_seq_len, dtype=ms.int32), "l -> l 3")

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    assert ref_H_tokens * ref_W_tokens == ref_img_len
                    # add image position ids

                    row_ids = repeat(mint.arange(ref_H_tokens, dtype=ms.int32), "h -> h w", w=ref_W_tokens).flatten()
                    col_ids = repeat(mint.arange(ref_W_tokens, dtype=ms.int32), "w -> h w", h=ref_H_tokens).flatten()
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            assert H_tokens * W_tokens == l_effective_img_len[i]

            row_ids = repeat(mint.arange(H_tokens, dtype=ms.int32), "h -> h w", w=W_tokens).flatten()
            col_ids = repeat(mint.arange(W_tokens, dtype=ms.int32), "w -> h w", h=H_tokens).flatten()

            assert pe_shift_len + l_effective_img_len[i] == seq_len
            position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len:seq_len, 1] = row_ids
            position_ids[i, pe_shift_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

        # create separate rotary embeddings for captions and images
        cap_freqs_cis = mint.zeros((batch_size, encoder_seq_len, freqs_cis.shape[-1]), dtype=freqs_cis.dtype)
        ref_img_freqs_cis = mint.zeros((batch_size, max_ref_img_len, freqs_cis.shape[-1]), dtype=freqs_cis.dtype)
        img_freqs_cis = mint.zeros((batch_size, max_img_len, freqs_cis.shape[-1]), dtype=freqs_cis.dtype)

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len, seq_lengths)
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[i, cap_seq_len : cap_seq_len + sum(ref_img_len)]
            img_freqs_cis[i, :img_len] = freqs_cis[
                i, cap_seq_len + sum(ref_img_len) : cap_seq_len + sum(ref_img_len) + img_len
            ]

        return cap_freqs_cis, ref_img_freqs_cis, img_freqs_cis, freqs_cis, l_effective_cap_len, seq_lengths
