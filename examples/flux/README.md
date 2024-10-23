# FLUX
introduced by Black Forest Labs: https://blackforestlabs.ai

adapted for MindSpore by @townwish



This example contains minimal inference code to run text-to-image and image-to-image with Black Forest Labs' Flux latent rectified flow transformers.


## Local installation

> [!NOTE]
> Our development and verification are based on the **MindSpore 2.3.1** version and the **CANN 7.3.0.1.231** version.

Install MindONE and dependencies for Flux.

```bash
cd $HOME && git clone https://github.com/mindspore-lab/mindone
cd $HOME/mindone
pip install -e .
pip install -r examples/flux/requirement.txt
```

To use Flux like the Black Forest Labs [official suggestion](#usage), you need to add the Flux path to the `PYTHONPATH` environment variable to ensure it can be called as a Python module.

```bash
export PYTHONPATH=<path_to_mindone>/examples:$PYTHONPATH
```

### Models

There are three models introduced by Black Forest Labs:
- `FLUX.1 [pro]` the base model, available via Black Forest Labs' API. **Not Available** in our repo
- `FLUX.1 [dev]` guidance-distilled variant
- `FLUX.1 [schnell]` guidance and step-distilled variant

| Name   | HuggingFace repo   | License    | md5sum    |
|-------------|-------------|-------------|-------------|
| `FLUX.1 [schnell]` | https://huggingface.co/black-forest-labs/FLUX.1-schnell | [apache-2.0](model_licenses/LICENSE-FLUX1-schnell) | a9e1e277b9b16add186f38e3f5a34044 |
| `FLUX.1 [dev]` | https://huggingface.co/black-forest-labs/FLUX.1-dev| [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) | a6bd8c16dfc23db6aee2f63a2eba78c0  |
| `FLUX.1 [pro]` | Only available in Black Forest Labs' API. **Not Available** in our repo  |

The weights of the autoencoder are also released under [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and can be found in either of the two HuggingFace repos above. They are the same for both models.


## Usage

The weights will be downloaded automatically from HuggingFace once you start one of the demos. To download `FLUX.1 [dev]`, you will need to be logged in, see [here](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login).
If you have downloaded the model weights manually, you can specify the downloaded paths via environment-variables:
```bash
export FLUX_SCHNELL=<path_to_flux_schnell_sft_file>
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
```

For interactive sampling run
```bash
python -m flux --name <name> --loop
```
Or to generate a single sample run
```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

Options:
- `--name`: Choose the model to use (options: "flux-schnell", "flux-dev")
- `--prompt`: Prompt used for sampling
- `--loop`: start an interactive session and sample multiple times

[Here](cli.py#L123) for more arguments usages.


## Limitations

### NSFW Classifier
As original Flux repo using `transformers.pipeline` to invoke the NSFW classifier, and adapting it to the MindSpore costs too much, we have abandoned support for NSFW filtering.

### `ModulationOut`
Since MindSpore's static graph syntax does not support returning instances of the `ModulationOut` class as return values of forward method, as seen in the original Flux implementation, we have replaced the `ModulationOut` class with tuples.

## TODO: Diffusers integration

`FLUX.1 [schnell]` and `FLUX.1 [dev]` will be integrated with the [mindone.diffusers](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers) library in mindone.diffusers v0.30 update.

Coming soon...
