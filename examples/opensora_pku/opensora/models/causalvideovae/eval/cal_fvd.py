from tqdm import tqdm

from mindspore import ops


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4)

    return x


def calculate_fvd(videos1, videos2, method="styleganv"):
    if method == "styleganv":
        from fvd.styleganv.fvd import frechet_distance, get_fvd_feats, load_i3d_pretrained
    elif method == "videogpt":
        from fvd.videogpt.fvd import frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from fvd.videogpt.fvd import load_i3d_pretrained

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained()
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}

    # for calculate FVD, each clip_timestamp must >= 10
    for clip_timestamp in tqdm(range(10, videos1.shape[-3] + 1)):
        # get a video clip
        # videos_clip [batch_size, channel, timestamps[:clip], h, w]
        videos_clip1 = videos1[:, :, :clip_timestamp]
        videos_clip2 = videos2[:, :, :clip_timestamp]

        # get FVD features
        feats1 = get_fvd_feats(
            videos_clip1,
            i3d=i3d,
        )
        feats2 = get_fvd_feats(
            videos_clip2,
            i3d=i3d,
        )

        # calculate FVD when timestamps[:clip]
        fvd_results[clip_timestamp] = frechet_distance(feats1, feats2)

    result = {
        "value": fvd_results,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, heigth, width",
    }

    return result


# test code / using example


def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = ops.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE)
    videos2 = ops.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE)

    import json

    result = calculate_fvd(videos1, videos2, method="videogpt")
    print(json.dumps(result, indent=4))

    result = calculate_fvd(videos1, videos2, method="styleganv")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
