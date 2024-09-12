import numpy as np
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from tqdm import tqdm


def trans(x):
    return x


def calculate_psnr(videos1, videos2):
    print("calculate_psnr...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    psnr_results = []

    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        psnr_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp]
            img2 = video2[clip_timestamp]

            # calculate psnr of a video
            psnr_results_of_a_video.append(cal_psnr(img1, img2))

        psnr_results.append(psnr_results_of_a_video)

    psnr_results = np.array(psnr_results)  # [batch_size, num_frames]
    psnr = {}
    psnr_std = {}

    for clip_timestamp in range(len(video1)):
        psnr[clip_timestamp] = np.mean(psnr_results[:, clip_timestamp])
        psnr_std[clip_timestamp] = np.std(psnr_results[:, clip_timestamp])

    result = {
        "value": psnr,
        "value_std": psnr_std,
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

    result = calculate_psnr(videos1, videos2)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
