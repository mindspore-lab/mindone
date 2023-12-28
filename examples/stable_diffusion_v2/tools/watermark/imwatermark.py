import argparse
import base64
import os
import struct
import uuid

import numpy as np
from PIL import Image
from tools._common import load_images
from tools.watermark.dwt_dct_svd import EmbedDwtDctSvd
from tools.watermark.max_dct import EmbedMaxDct

import mindspore as ms
from mindspore import Tensor, ops

WATERMARK_MESSAGE = "StableDiffusion"


class WatermarkEncoder(object):
    def __init__(self, content=b""):
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)
        self._wmType = "bytes"

    def set_by_ipv4(self, addr):
        bits = []
        ips = addr.split(".")
        for ip in ips:
            bits += list(np.unpackbits(np.array([ip % 255], dtype=np.uint8)))
        self._watermarks = bits
        self._wmLen = len(self._watermarks)
        self._wmType = "ipv4"
        assert self._wmLen == 32

    def set_by_uuid(self, uid):
        u = uuid.UUID(uid)
        self._wmType = "uuid"
        seq = np.array([n for n in u.bytes], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_bytes(self, content):
        self._wmType = "bytes"
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_b16(self, b16):
        content = base64.b16decode(b16)
        self.set_by_bytes(content)
        self._wmType = "b16"

    def set_by_bits(self, bits=[]):
        self._watermarks = [int(bit) % 2 for bit in bits]
        self._wmLen = len(self._watermarks)
        self._wmType = "bits"

    def set_watermark(self, wmType="bytes", content=""):
        if wmType == "ipv4":
            self.set_by_ipv4(content)
        elif wmType == "uuid":
            self.set_by_uuid(content)
        elif wmType == "bits":
            self.set_by_bits(content)
        elif wmType == "bytes":
            self.set_by_bytes(content)
        elif wmType == "b16":
            self.set_by_b16(content)
        else:
            raise NameError("%s is not supported" % wmType)

    def get_length(self):
        return self._wmLen

    def encode(self, cv2Image, method="dwtDct", **configs):
        (r, c, channels) = cv2Image.shape
        if r * c < 256 * 256:
            raise RuntimeError("image too small, should be larger than 256x256")

        if method == "dwtDct":
            embed = EmbedMaxDct(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == "dwtDctSvd":
            embed = EmbedDwtDctSvd(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        else:
            raise NameError("%s is not supported" % method)


class WatermarkDecoder(object):
    def __init__(self, wm_type="bytes", length=0):
        self._wmType = wm_type
        if wm_type == "ipv4":
            self._wmLen = 32
        elif wm_type == "uuid":
            self._wmLen = 128
        elif wm_type == "bytes":
            self._wmLen = length
        elif wm_type == "bits":
            self._wmLen = length
        elif wm_type == "b16":
            self._wmLen = length
        else:
            raise NameError("%s is unsupported" % wm_type)

    def reconstruct_ipv4(self, bits):
        ips = [str(ip) for ip in list(np.packbits(bits))]
        return ".".join(ips)

    def reconstruct_uuid(self, bits):
        nums = np.packbits(bits)
        bstr = b""
        for i in range(16):
            bstr += struct.pack(">B", nums[i])

        return str(uuid.UUID(bytes=bstr))

    def reconstruct_bits(self, bits):
        # return ''.join([str(b) for b in bits])
        return bits

    def reconstruct_b16(self, bits):
        bstr = self.reconstruct_bytes(bits)
        return base64.b16encode(bstr)

    def reconstruct_bytes(self, bits):
        nums = np.packbits(bits)
        bstr = b""
        for i in range(self._wmLen // 8):
            bstr += struct.pack(">B", nums[i])
        return bstr

    def reconstruct(self, bits):
        if len(bits) != self._wmLen:
            raise RuntimeError("bits are not matched with watermark length")

        if self._wmType == "ipv4":
            return self.reconstruct_ipv4(bits)
        elif self._wmType == "uuid":
            return self.reconstruct_uuid(bits)
        elif self._wmType == "bits":
            return self.reconstruct_bits(bits)
        elif self._wmType == "b16":
            return self.reconstruct_b16(bits)
        else:
            return self.reconstruct_bytes(bits)

    def decode(self, cv2Image, method="dwtDct", **configs):
        (r, c, channels) = cv2Image.shape
        if r * c < 256 * 256:
            raise RuntimeError("image too small, should be larger than 256x256")

        bits = []
        if method == "dwtDct":
            embed = EmbedMaxDct(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == "dwtDctSvd":
            embed = EmbedDwtDctSvd(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        else:
            raise NameError("%s is not supported" % method)
        return self.reconstruct(bits)


class WatermarkEmbedder:
    def __init__(self, dtype=ms.float16):
        self.watermark = WATERMARK_MESSAGE
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bytes", self.watermark.encode("utf-8"))
        self.dtype = dtype
        self.decoder = WatermarkDecoder("bytes", 120)

    def __call__(self, images):
        # can't encode images that are smaller than 256
        if images.shape[-1] < 256:
            return images
        images_ = []
        for i in range(images.shape[0]):
            im = ((255.0 * images[i].transpose(1, 2, 0))).astype(np.uint8).asnumpy()
            im = self.encoder.encode(im, "dwtDct")
            im = Tensor(np.asarray(im), self.dtype)
            images_.append(im)
        images = ops.stack(images_).transpose((0, 3, 1, 2))

        images = ops.clip_by_value((images / 255), clip_value_min=0.0, clip_value_max=1.0)

        return images

    def add_watermark(self, path):
        images, paths = load_images(path, resize=False)
        base_count = len(os.listdir(path))
        print(f"{len(images)} images are loaded")
        for image in images:
            im = np.array(image)
            im = self.encoder.encode(im, "dwtDct")
            images = Image.fromarray(im.astype(np.uint8))
            images.save(os.path.join(path, f"{base_count:05}.png"))
            base_count += 1

    def decoder_watermark(self, path):
        images, paths = load_images(path, resize=False)

        print(f"{len(images)} images are loaded")
        for image in images:
            im = np.array(image)
            im = self.decoder.decode(im, "dwtDct")
            try:
                res = im.decode("utf-8")
            except UnicodeDecodeError:
                res = "null"
            print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Directory to images")
    parser.add_argument(
        "--watermark_name",
        default="encoder",
        type=str,
        help="encoder watermark or decoder watermark. Option: encoder, decoder. Default: encoder",
    )
    args = parser.parse_args()
    watermark = WatermarkEmbedder()
    if args.watermark_name == "encoder":
        images = watermark.add_watermark(args.image_path)
    elif args.watermark_name == "decoder":
        images = watermark.decoder_watermark(args.image_path)
    else:
        raise ValueError(f"Unknown watermark_name: {args.watermark_name}. Valid watermark_name: [encoder, decoder]")
    print("done")
