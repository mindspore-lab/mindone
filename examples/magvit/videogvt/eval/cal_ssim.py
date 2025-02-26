import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def calculate_ssim_function(img1, img2):
    # [0,1]
    # ssim is the only metric extremely sensitive to gray being compared to b/w
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def trans(x):
    return x


def calculate_ssim(videos1, videos2):
    print("calculate_ssim...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    ssim_results = []

    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        ssim_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp]
            img2 = video2[clip_timestamp]

            # calculate ssim of a video
            ssim_results_of_a_video.append(calculate_ssim_function(img1, img2))

        ssim_results.append(ssim_results_of_a_video)

    ssim_results = np.array(ssim_results)

    ssim = {}
    ssim_std = {}

    for clip_timestamp in range(len(video1)):
        ssim[clip_timestamp] = np.mean(ssim_results[:, clip_timestamp])
        ssim_std[clip_timestamp] = np.std(ssim_results[:, clip_timestamp])

    result = {
        "value": ssim,
        "value_std": ssim_std,
        "video_setting": video1.shape,
        "video_setting_name": "time, channel, heigth, width",
    }

    return result


# test code / using example


def main():
    from mindspore import ops

    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = ops.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = ops.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)

    import json

    result = calculate_ssim(videos1, videos2)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
