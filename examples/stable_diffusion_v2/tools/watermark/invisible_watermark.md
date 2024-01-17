# Invisible Watermark

invisible watermark is a tool for creating invisible watermark over image.

## Usage

You can use the invisible watermark in your image generation pipelines. It is by default included in `text_to_image.py`
for adding invisible watermark to the stable diffusion output images. In addition, you can encode or decode watermark on saved images.
In `examples/stable_diffusion_v2` directory, please run

```
python tools/water/imwatermark.py --image_path {dir_to_images}  ----watermark_name {endcoder_or_decoder}
```

> This algorithm cannot guarantee 100% accuracy in decoding the original watermark.



## Reference

[1] https://github.com/ShieldMnt/invisible-watermark/
