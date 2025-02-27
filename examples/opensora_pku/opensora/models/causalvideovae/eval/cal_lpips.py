import os

import numpy as np
from opensora.models.causalvideovae.model.losses.lpips import LPIPS

import mindspore as ms
from mindspore import mint

spatial = True  # Return a spatial map of perceptual distance.
lpips_ckpt_path = os.path.join("pretrained", "lpips_vgg-426bf45c.ckpt")
# Linearly calibrated models (LPIPS)
loss_fn = LPIPS()  # freeze params inside
assert os.path.exists(lpips_ckpt_path), (
    f"LPIPS ckpt path {lpips_ckpt_path} is not existent. "
    + "Please download it from https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt and put it under pretrained/"
)
loss_fn.load_from_pretrained(lpips_ckpt_path)
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'
print("Calculating LPIPS loss using VGG16.")


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(3, axis=2)

    # value range [0, 1] -> [-1, 1]
    x = x * 2 - 1

    return x


def calculate_lpips(videos1, videos2):
    # image should be RGB, IMPORTANT: normalized to [-1,1]

    assert videos1.shape == videos2.shape

    # videos [batch_size, timestamps, channel, h, w]

    # support grayscale input, if grayscale -> channel*3
    # value range [0, 1] -> [-1, 1]
    videos1 = trans(videos1)
    videos2 = trans(videos2)

    lpips_results = []

    for video_num in range(videos1.shape[0]):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        video1 = ms.Tensor(video1, dtype=ms.float32)
        video2 = ms.Tensor(video2, dtype=ms.float32)

        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp].unsqueeze(0)
            img2 = video2[clip_timestamp].unsqueeze(0)

            # calculate lpips of a video
            lpips_results_of_a_video.append(loss_fn(img1, img2).mean().asnumpy().tolist())
        lpips_results.append(lpips_results_of_a_video)

    lpips_results = np.array(lpips_results)

    lpips = {}
    lpips_std = {}

    for clip_timestamp in range(len(video1)):
        lpips[clip_timestamp] = np.mean(lpips_results[:, clip_timestamp])
        lpips_std[clip_timestamp] = np.std(lpips_results[:, clip_timestamp])

    result = {
        "value": lpips,
        "value_std": lpips_std,
        "video_setting": video1.shape,
        "video_setting_name": "time, channel, heigth, width",
    }

    return result


# test code / using example


def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = mint.zeros(
        NUMBER_OF_VIDEOS,
        VIDEO_LENGTH,
        CHANNEL,
        SIZE,
        SIZE,
    )
    videos2 = mint.ones(
        NUMBER_OF_VIDEOS,
        VIDEO_LENGTH,
        CHANNEL,
        SIZE,
        SIZE,
    )

    import json

    result = calculate_lpips(videos1, videos2)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
