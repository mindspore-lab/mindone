# Compel

A MindSpore implementation of the [Compel](https://github.com/damian0815/compel) Library, used for text prompt weighting and blending.

- [x] support SDV2(2.1) and SDV1.5 text prompt re-weighting.
- [x] support returning last layer's output after normalization as the text embedding.
- [ ] support SD-XL text prompt reweighting.


## Dependency

- mindspore >=2.1.0
- transformers >=4.30

## QuickStart

`text_to_image_compel.py` is a variant of `examples/stable_diffusion_v2/text_to_image.py` which allows using `Compel` to reweight text prompts by default. Check `--use_compel` in `text_to_image_compel.py` for more details.

Please change the current working directory to `examples/stable_diffusion_v2/`. Then run the following command to generate an image of a cat without using `Compel`:
```
python ../compel/text_to_image_compel.py --use_compel False --version "2.0" --prompt "A cute cat on the grass wearing a blue hat"
```

If we want to increase the weight of "hat", we can
