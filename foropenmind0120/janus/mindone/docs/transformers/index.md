# Get Pretrained Txt/Img Encoder from 🤗 Transformers


This MindSpore patch for [🤗 Transformers](https://github.com/huggingface/transformers) enables researchers or developers
in the field of text-to-image (t2i) and text-to-video (t2v) generation to utilize pretrained text and image models from 🤗 Transformers on MindSpore.
The pretrained models from 🤗 Transformers can be employed either as frozen encoders or fine-tuned with denoising networks for generative tasks.
This approach **_aligns with the practices_** of PyTorch users<sup>[[1]](https://github.com/huggingface/diffusers)[[2]](https://github.com/Stability-AI/generative-models)</sup>.
Now, MindSpore users can benefit from the same functionality!

## Philosophy

- Only the MindSpore model definition will be implemented, which will be identical to the PyTorch model.
- Configuration, Tokenizer, etc. will utilize the original 🤗 Transformers.
- Models here will be limited to the scope of generative tasks.
