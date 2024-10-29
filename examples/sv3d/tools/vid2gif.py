import imageio
import numpy as np
import PIL.Image as Image


def video2gif(video_p, gif_p, fps=10):
    r = imageio.v2.mimread(video_p)
    r = np.array(r)
    img = Image.fromarray(r)
    img[0].save(
        gif_p,
        save_all=True,
        append_images=img[1:],
        optimize=False,
        duration=1000 // fps,
        loop=0,
    )


class DumperGif(object):
    def __init__(self, fps=7):
        self.fps = fps

    def vid2gif(self, gif_p, img_arr):
        img = [Image.fromarray(i) for i in img_arr]
        img[0].save(
            gif_p,
            save_all=True,
            append_images=img[1:],
            optimize=False,
            duration=1000 // self.fps,
            loop=0,
        )
