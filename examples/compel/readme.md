# Compel

A MindSpore implementation of the [Compel](https://github.com/damian0815/compel) Library, used for text prompt weighting and blending.

- [x] support SDV2(2.1) and SDV1.5 text prompt re-weighting.
- [x] support returning last layer's output after normalization as the text embedding.
- [ ] support SD-XL text prompt reweighting.


## Dependency

- mindspore >=2.1.0
- transformers >=4.30

## QuickStart for SDv2 (SDv1.5)

`text_to_image_compel.py` is a variant of `examples/stable_diffusion_v2/text_to_image.py` which allows using `Compel` to reweight text prompts by default. Check `--use_compel` in `text_to_image_compel.py` for more details.

Please change the current working directory to `examples/stable_diffusion_v2/`. Then run the following command to generate an image of a cat without using `Compel`:
```
python text_to_image.py --version "2.0" --prompt "a cute wolf running in the winter forest wearing a blue hat"
```

The generated image is shown below, in which the wolf is not wearing a blue hat.
<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/compel/wolf.png" width=450 />
</p>

If we want to increase the weight of "blue hat", we can use the following command:

```
python ../compel/text_to_image_compel.py --version "2.0" --prompt "a cute wolf running in the winter forest wearing a blue+++ hat+++"
```

- `+` corresponds to the value 1.1, `++` corresponds to 1.1^2, and so on. Similarly, `-` corresponds to 0.9 and `--` corresponds to 0.9^2.

This generates the image shown below:
<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/compel/blue_hat_plus.png" width=450 />
</p>

We can further decrease the weight of "winter" using this command:
```
python ../compel/text_to_image_compel.py --version "2.0" --prompt "a cute wolf running in the winter------- forest with a blue+++ hat+++"
```

The generated image is like this:
<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/compel/winter_minus.png" width=450 />
</p>
