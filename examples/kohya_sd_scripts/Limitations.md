# Limitations

We've tried to provide a consistent implementation with the torch Kohya SD trainer. Due to differences in framework, we have made the necessary changes to the MindSpore version. Here we list the main differences between the two codebases.


## 1. `network.lora`

The conception `network` in Kohya means the respective LoRA network of the target tuning net, such as SD's UNet.

In torch Kohya, the `LoRANetwork` is made of `LoRAModule`. Once the `apply_to` method is called, torch Kohya replaces forward method of the original Linear from the target modules of `UNet`, instead of dircectly replacing the original Linear module. So basically the `UNet` and the `LoRANetwork` are two independent `nn.Module`, and some linear layers of `UNet` use the respective forward methods of `LoRAModule` in  `LoRANetwork` when computing.

In MindSpore graph mode, forward method replacements raise errors in graph construction. The respective linear layers from original net and the added lora layers must in the same subgraph. Thus the MindSpore implementation replaces the original linear module of `UNet` when creating the`LoRANetwork`, and the `UNet` as a `nn.Cell` is packaged by the `LoRANetwork` as a bigger `nn.Cell`.


### class - `LoRANetwork`

*Torch*

```python
class LoRANetwork(torch.nn.Module):
    ...
    def __init__(...):
        ...
        # func for creating <LoRAMoudle> instances
        def create_modules(...):
            ...
        # search unet and text encoders TARGET_REPLACE_MODULE and create respective LoRA modules
        self.unet_loras = create_modules(...)
        self.text_encoder_loras = create_modules(...)

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        ...
        for lora in self.text_encoder_loras + self.unet_loras:
            # apply the forward method of original modules
            lora.apply_to()
            # incert the <LoRAMoudle> to the <LoRANetwork>
            self.add_module(lora.lora_name, lora)

    # for inference
    def merge_to(self, weights_sd):
        ...
        for lora in self.text_encoder_loras + self.unet_loras:
            ...
            lora.merge_to(weights_sd)
```

*Mindspore*

```python
class LoRANetwork(nn.Cell):
    ...
    def __init__(...):
        ...
        # unet and text encoders are packaged in `LoRANetwork` now
        self.text_encoders = text_encoders
        self.text_encoders = unet

        # modified from `create_modules` method, but more than creation, once we find the target we replace the module
        def replace_modules(...):
            ...
        # search unet and text encoders TARGET_REPLACE_MODULE and replace respective LoRA modules
        replace_modules(unet)
        replace_modules(text_encoders)
        ...
    # no apply_to and merge_to method here
```


### APIs - `create_network` and `create_network_from_weights`

The two APIs are for lora network creation in training and inference scripts. Torch use create_network to initalize a `LoRANetwork` and call `LoRANetwork.apply_to` to replace the forward method as said above. MindSpore creates and replaces the modules in initailztion and no need to have the apply method. Similarly in inference, Torch creates the `LoRANetwork` first and call the `LoRANetwork.merge_to` for lora weights loading and merging. MindSpore directly loads and merges the weights to unet or text encoders by create_network_from_weights API, without any network creation.

*Torch*

```python
# from train_network.py NetworkTrainer.train
network_module = importlib.import_module("networks.lora")
network = network_module.create_network(...)
network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

# from sdxl_minimal_inference.py
for weights_file in args.lora_weights:
    ...
    lora_model, weights_sd = lora.create_network_from_weights(
        multiplier, weights_file, vae, [text_model1, text_model2], unet, None, True
    )
    lora_model.merge_to([text_model1, text_model2], unet, weights_sd, DTYPE, DEVICE)
```

*MindSpore*

```python
# from train_network.py NetworkTrainer.train
network_module = importlib.import_module("networks.lora")
network = network_module.create_network(...)
# no apply_to here

# from sdxl_minimal_inference.py
for weights_file in args.lora_weights:
    ...
    # lora weights are directly loaded and merged here
    lora.create_network_from_weights(
        multiplier, weights_file, vae, [text_model1, text_model2], unet, weights_sd=None, dtype=DTYPE
    )
```


## 2. inputs and outputs of methods in `SdxlNetworkTrainer`

MindSpore graph mode may raise errors in graph construction for dictionary inputs or outputs. Thus the`load_target_model`, `load_tokenizer`, `get_text_cond` , `call_unet` methods in `SdxlNetworkTrainer` or any other util related to graph compile use tuple inputs or outputs.

For example,

*Torch*

```python
# from train_network.py NetworkTrainer.train
class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype
        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        ...
        return xxx
```

*MindSpore*

```python
# from SDXLNetworkTrainer.call_unet in sdxl_train_network.py
class SdxlNetworkTrainer(train_network.NetworkTrainer):
    # specify the values in `batch`
    def call_unet(self, args, unet, noisy_latents, timesteps, text_conds, original_sizes_hw, crop_top_lefts, target_sizes_hw, weight_dtype,
  ):
    ...
    return xxx
```


## 3. forward and backward wrapper

Kohya designs a base lora trainer in `NetworkTrainer` in `train_network.py`,  with `load_target_model`, `load_tokenizer`, `get_text_cond`, `call_unet`, and other methods. Most of these methods are designed for SD training and are overridden in`SDXLNetworkTrainer`, except the `train` method. The `NetworkTrainer.train` in torch kohya implements a general lora training for the whole process including dataset, network, optimizer, forward and backward.

Different from Torch, automatic differentiation in MindSpore graph mode based on the graph structure of forward graphs and backward graphs. So we define a `TrainStepForSDXLLoRA` to wrap the forward and backward process, and the `train` method needs to choose the trainstep wrapper for any specified model in training loop. That means another model like SD1.5 or Flux needs another trainstep wrapper definition. The wrapper inherits the `TrainStep` from `mindone.diffusers.training_utils`, a base class for training steps implemented in MindSpore.

*Torch*

```python
# training loop
for step, batch in enumerate(train_dataloader):

    # define the forward process
    ...
    latents = ...
    text_conds = ..
    noise_pred = self.call_unet(batch, ...)
    ...
    # compute loss
    loss = ...

    # backward
    accelerator.backward(loss)
    ...

```

*MindSpore*

```python
# training loop
train_step = train_step = self.train_step_class(...) # TrainStepForSDXLLoRA
for step, batch in enumerate(train_dataloader_iter):
    ...
    inputs = (
        batch["loss_weights"],
        batch["input_ids"],
        batch["input_ids2"],
        batch["images"],
        batch["original_sizes_hw"],
        batch["crop_top_lefts"],
        batch["target_sizes_hw"],
    )
    loss, model_pred = train_step(*inputs)

# define the TrainStepForSDXLLoRA
from mindone.diffusers.training_utils import TrainStep
class TrainStepForSDXLLoRA(TrainStep):
    def __init__(self,..):
        ...
    def forward(self...):
        # override the forward method of `TrainStep`
        # define the forward process
        ...
        latents = ...
        text_conds = ..
        noise_pred = self.call_unet(batch...)
        ...
    # the backward is wrpped in `TrainStep`, invisible here.
    # refer to `TrainStep` if needed.
```
