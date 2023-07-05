# 1. Stable Diffusion 2.0

## 1.1 Inference

Step 1. Download the [SD2.0 checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) and put it under `models/` folder

Step 2. Run `text_to_image.py` to generate images for the prompt of your interest.


```python
# Stable Diffusion 2.0 Inference
!python text_to_image.py --prompt 'A cute wolf in winter forest'
```

    workspace /home/yx/mindone/examples/stable_diffusion_v2
    WORK DIR:/home/yx/mindone/examples/stable_diffusion_v2
    Loading model from models/sd_v2_base-57526ee4.ckpt
    LatentDiffusion: Running in eps-prediction mode
    Attention: output_channels=1280, num_heads=20, dim_head=64
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    Using tokenizer `BpeTokenizer` for en.
    [WARNING] ME(1465704:140139123439424,MainProcess):2023-06-09-15:39:37.864.135 [mindspore/train/serialization.py:1058] For 'load_param_into_net', 4 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict', please check whether the network structure is consistent when training and loading checkpoint.
    [WARNING] ME(1465704:140139123439424,MainProcess):2023-06-09-15:39:37.864.262 [mindspore/train/serialization.py:1060] first_stage_model.encoder.down.3.downsample.conv.weight is not loaded.
    [WARNING] ME(1465704:140139123439424,MainProcess):2023-06-09-15:39:37.864.294 [mindspore/train/serialization.py:1060] first_stage_model.encoder.down.3.downsample.conv.bias is not loaded.
    [WARNING] ME(1465704:140139123439424,MainProcess):2023-06-09-15:39:37.864.318 [mindspore/train/serialization.py:1060] first_stage_model.decoder.up.0.upsample.conv.weight is not loaded.
    [WARNING] ME(1465704:140139123439424,MainProcess):2023-06-09-15:39:37.864.340 [mindspore/train/serialization.py:1060] first_stage_model.decoder.up.0.upsample.conv.bias is not loaded.
    param not load: (['first_stage_model.encoder.down.3.downsample.conv.weight', 'first_stage_model.encoder.down.3.downsample.conv.bias', 'first_stage_model.decoder.up.0.upsample.conv.weight', 'first_stage_model.decoder.up.0.upsample.conv.bias'], ['cond_stage_model.transformer.transformer_layer.resblocks.23.attn.attn.in_proj.bias', 'cond_stage_model.transformer.transformer_layer.resblocks.23.attn.attn.in_proj.weight', 'cond_stage_model.transformer.transformer_layer.resblocks.23.attn.attn.out_proj.bias', 'cond_stage_model.transformer.transformer_layer.resblocks.23.attn.attn.out_proj.weight', 'cond_stage_model.transformer.transformer_layer.resblocks.23.c_fc.bias', 'cond_stage_model.transformer.transformer_layer.resblocks.23.c_fc.weight', 'cond_stage_model.transformer.transformer_layer.resblocks.23.c_proj.bias', 'cond_stage_model.transformer.transformer_layer.resblocks.23.c_proj.weight', 'cond_stage_model.transformer.transformer_layer.resblocks.23.ln_1.beta', 'cond_stage_model.transformer.transformer_layer.resblocks.23.ln_1.gamma', 'cond_stage_model.transformer.transformer_layer.resblocks.23.ln_2.beta', 'cond_stage_model.transformer.transformer_layer.resblocks.23.ln_2.gamma'])
    Data shape for PLMS sampling is (8, 4, 64, 64)
    Running PLMS Sampling with 50 timesteps
    the infer time of a batch is 140.39605832099915
    Data shape for PLMS sampling is (8, 4, 64, 64)
    Running PLMS Sampling with 50 timesteps
    the infer time of a batch is 24.638887643814087
    Your samples are ready and waiting for you here: 
    output 
     
    Enjoy.


> Note: The SD2.0 checkpoint does NOT well support Chinese prompts. If you prefer to use Chinese prompts, please refer to Section 2.1.


```python
# The generated images are saved in `output/samples` folder by default
!ls output/samples
```

    00000.png  00003.png  00006.png  00009.png  00012.png  00015.png
    00001.png  00004.png  00007.png  00010.png  00013.png
    00002.png  00005.png  00008.png  00011.png  00014.png



```python
# let's see what it makes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('output/samples/00012.png')
imgplot = plt.imshow(img)
plt.show()
```


![output_4_0](https://github.com/SamitHuang/mindone/assets/8156835/65fc481a-cbc9-4000-b1b7-94875ef76e43)
    


## 1.2 SD2.0 Finetune (Vanilla)

Step 1. Download the [SD2.0 checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) and put it under `models/` folder

Step 2. Prepare your image-text pair data (referring to README.md) and change `data_path` in `scripts/run_train_v2.sh` accordingly.

Step 3. Run the training script as follows


```python
# After preparing the checkpoint and image-text pair data, run the follow script to finetune SD2.0 on a single NPU

!sh scripts/run_train_v2.sh
```

    process id: 1488387
    Namespace(betas=[0.9, 0.98], callback_size=1, data_path='/home/yx/datasets/diffusion/pokemon', decay_steps=0, dropout=0.1, end_learning_rate=1e-07, epochs=20, filter_small_size=True, gradient_accumulation_steps=1, image_filter_size=256, image_size=512, init_loss_scale=65536, loss_scale_factor=2, model_config='/home/yx/mindone/examples/stable_diffusion_v2/configs/v2-train-chinese.yaml', optim='adamw', output_path='output//txt2img', patch_size=32, pretrained_model_file='sd_v2_base-57526ee4.ckpt', pretrained_model_path='models/', random_crop=False, save_checkpoint_steps=10000, scale_window=1000, seed=3407, start_learning_rate=1e-05, train_batch_size=3, train_config='/home/yx/mindone/examples/stable_diffusion_v2/configs/train_config_v2.json', use_parallel=False, warmup_steps=1000, weight_decay=0.01)
    random seed:  3407
    Filter small images, filter size: 256
    The first image path is /home/yx/datasets/diffusion/pokemon/img1.jpg, and the caption is a cartoon character with a potted plant on his head
    total data num: 3
    rank id 0, sample num is 1
    LatentDiffusion: Running in eps-prediction mode
    Attention: output_channels=1280, num_heads=20, dim_head=64
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    Using tokenizer `BpeTokenizer` for en.
    start loading pretrained_ckpt models/sd_v2_base-57526ee4.ckpt
    param not load: (['first_stage_model.encoder.down.3.downsample.conv.weight', 'first_stage_model.encoder.down.3.downsample.conv.bias', 'first_stage_model.decoder.up.0.upsample.conv.weight', 'first_stage_model.decoder.up.0.upsample.conv.bias'], ['cond_stage_model.transformer.transformer_layer.resblocks.23.attn.attn.in_proj.bias', 'cond_stage_model.transformer.transformer_layer.resblocks.23.attn.attn.in_proj.weight', 'cond_stage_model.transformer.transformer_layer.resblocks.23.attn.attn.out_proj.bias', 'cond_stage_model.transformer.transformer_layer.resblocks.23.attn.attn.out_proj.weight', 'cond_stage_model.transformer.transformer_layer.resblocks.23.c_fc.bias', 'cond_stage_model.transformer.transformer_layer.resblocks.23.c_fc.weight', 'cond_stage_model.transformer.transformer_layer.resblocks.23.c_proj.bias', 'cond_stage_model.transformer.transformer_layer.resblocks.23.c_proj.weight', 'cond_stage_model.transformer.transformer_layer.resblocks.23.ln_1.beta', 'cond_stage_model.transformer.transformer_layer.resblocks.23.ln_1.gamma', 'cond_stage_model.transformer.transformer_layer.resblocks.23.ln_2.beta', 'cond_stage_model.transformer.transformer_layer.resblocks.23.ln_2.gamma'])
    end loading ckpt
    start_training...
    epoch: 1 step: 1, loss is 0.18253887
    Train epoch time: 412761.042 ms, per step time: 412761.042 ms
    epoch: 2 step: 1, loss is 0.034258194
    Train epoch time: 1518.288 ms, per step time: 1518.288 ms
    epoch: 3 step: 1, loss is 0.08877602
    Train epoch time: 1498.360 ms, per step time: 1498.360 ms
    epoch: 4 step: 1, loss is 0.045948993
    Train epoch time: 1499.529 ms, per step time: 1499.529 ms
    epoch: 5 step: 1, loss is 0.09729623
    Train epoch time: 1490.405 ms, per step time: 1490.405 ms
    epoch: 6 step: 1, loss is 0.045910917
    Train epoch time: 1487.570 ms, per step time: 1487.570 ms
    epoch: 7 step: 1, loss is 0.18001567
    Train epoch time: 1493.911 ms, per step time: 1493.911 ms
    epoch: 8 step: 1, loss is 0.074366115
    Train epoch time: 1484.088 ms, per step time: 1484.088 ms
    epoch: 9 step: 1, loss is 0.042577006
    Train epoch time: 1485.518 ms, per step time: 1485.518 ms
    epoch: 10 step: 1, loss is 0.1435442
    Train epoch time: 1499.919 ms, per step time: 1499.919 ms
    epoch: 11 step: 1, loss is 0.19214934
    Train epoch time: 1489.204 ms, per step time: 1489.204 ms
    epoch: 12 step: 1, loss is 0.11975494
    Train epoch time: 1347.991 ms, per step time: 1347.991 ms
    epoch: 13 step: 1, loss is 0.15481588
    Train epoch time: 1340.607 ms, per step time: 1340.607 ms
    epoch: 14 step: 1, loss is 0.076101474
    Train epoch time: 1346.508 ms, per step time: 1346.508 ms
    epoch: 15 step: 1, loss is 0.12587808
    Train epoch time: 1346.990 ms, per step time: 1346.990 ms
    epoch: 16 step: 1, loss is 0.1882276
    Train epoch time: 1342.456 ms, per step time: 1342.456 ms
    epoch: 17 step: 1, loss is 0.08895156
    Train epoch time: 1342.833 ms, per step time: 1342.833 ms
    epoch: 18 step: 1, loss is 0.082536414
    Train epoch time: 1343.141 ms, per step time: 1343.141 ms
    epoch: 19 step: 1, loss is 0.035138734
    Train epoch time: 1343.694 ms, per step time: 1343.694 ms
    epoch: 20 step: 1, loss is 0.14512514
    Train epoch time: 1344.402 ms, per step time: 1344.402 ms


# 2. Stable Diffusion 1.x (Chinese)

## 2.1 Inference

Step 1. Download the [SD1.x checkpoint](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) (credit to WuKongHuaHua) and put it under `models/` folder

Step 2. Run `text_to_image.py` to generate images for the prompt of your interest and specify the `-v` arg.



```python
# Stable Diffusion 1.x Inference 
!python text_to_image.py --prompt '雪中之狼' -v 1.x
```

    workspace /home/yx/mindone/examples/stable_diffusion_v2
    WORK DIR:/home/yx/mindone/examples/stable_diffusion_v2
    Loading model from models/wukong-huahua-ms.ckpt
    LatentDiffusion: Running in eps-prediction mode
    Attention: output_channels=1280, num_heads=8, dim_head=-1
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    Using tokenizer `WordPieceTokenizer` for zh.
    [WARNING] ME(1541416:139846282307392,MainProcess):2023-06-09-16:05:43.851.136 [mindspore/train/serialization.py:167] The type of cond_stage_model.transformer.embedding_table:Float32 in 'parameter_dict' is different from the type of it in 'net':Float16, then the type convert from Float32 to Float16 in the network.
    [WARNING] ME(1541416:139846282307392,MainProcess):2023-06-09-16:05:43.937.897 [mindspore/train/serialization.py:167] The type of cond_stage_model.transformer.positional_embedding:Float32 in 'parameter_dict' is different from the type of it in 'net':Float16, then the type convert from Float32 to Float16 in the network.
    [WARNING] ME(1541416:139846282307392,MainProcess):2023-06-09-16:05:44.218.411 [mindspore/train/serialization.py:1058] For 'load_param_into_net', 4 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict', please check whether the network structure is consistent when training and loading checkpoint.
    [WARNING] ME(1541416:139846282307392,MainProcess):2023-06-09-16:05:44.218.517 [mindspore/train/serialization.py:1060] first_stage_model.encoder.down.3.downsample.conv.weight is not loaded.
    [WARNING] ME(1541416:139846282307392,MainProcess):2023-06-09-16:05:44.218.545 [mindspore/train/serialization.py:1060] first_stage_model.encoder.down.3.downsample.conv.bias is not loaded.
    [WARNING] ME(1541416:139846282307392,MainProcess):2023-06-09-16:05:44.218.568 [mindspore/train/serialization.py:1060] first_stage_model.decoder.up.0.upsample.conv.weight is not loaded.
    [WARNING] ME(1541416:139846282307392,MainProcess):2023-06-09-16:05:44.218.590 [mindspore/train/serialization.py:1060] first_stage_model.decoder.up.0.upsample.conv.bias is not loaded.
    param not load: (['first_stage_model.encoder.down.3.downsample.conv.weight', 'first_stage_model.encoder.down.3.downsample.conv.bias', 'first_stage_model.decoder.up.0.upsample.conv.weight', 'first_stage_model.decoder.up.0.upsample.conv.bias'], [])
    Data shape for PLMS sampling is (8, 4, 64, 64)
    Running PLMS Sampling with 50 timesteps
    the infer time of a batch is 133.70874071121216
    Data shape for PLMS sampling is (8, 4, 64, 64)
    Running PLMS Sampling with 50 timesteps
    the infer time of a batch is 30.258379220962524
    Your samples are ready and waiting for you here: 
    output 
     
    Enjoy.



```python
# The generated images are saved in `output/samples` folder by default
!ls output/samples
```

    00000.png  00005.png  00010.png  00015.png  00020.png  00025.png  00030.png
    00001.png  00006.png  00011.png  00016.png  00021.png  00026.png  00031.png
    00002.png  00007.png  00012.png  00017.png  00022.png  00027.png
    00003.png  00008.png  00013.png  00018.png  00023.png  00028.png
    00004.png  00009.png  00014.png  00019.png  00024.png  00029.png



```python
img = mpimg.imread('output/samples/00030.png')
imgplot = plt.imshow(img)
plt.show()
```


![output_10_0](https://github.com/SamitHuang/mindone/assets/8156835/593f8a80-ca47-4316-9ad2-86e545d3dc62)

## 2.2 SD1.x Finetune (Vanilla)

Step 1.Download the [SD1.x checkpoint](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) (credit to WuKongHuaHua) and put it under `models/` folder

Step 2. Prepare your image-text pair data (referring to README.md) and change `data_path` in `scripts/run_train_v1.sh` accordingly.

Step 3. Run the training script as follows


```python
# After preparing the checkpoint and image-text pair data, run the follow script to finetune SD2.0 on a single NPU

!sh scripts/run_train_v1.sh
```

    process id: 1561980
    Namespace(betas=[0.9, 0.98], callback_size=1, data_path='/home/yx/datasets/diffusion/pokemon', decay_steps=0, dropout=0.1, end_learning_rate=1e-07, epochs=20, filter_small_size=True, gradient_accumulation_steps=1, image_filter_size=256, image_size=512, init_loss_scale=65536, loss_scale_factor=2, model_config='/home/yx/mindone/examples/stable_diffusion_v2/configs/v1-train-chinese.yaml', optim='adamw', output_path='output//txt2img', patch_size=32, pretrained_model_file='wukong-huahua-ms.ckpt', pretrained_model_path='models/', random_crop=False, save_checkpoint_steps=10000, scale_window=1000, seed=3407, start_learning_rate=1e-05, train_batch_size=3, train_config='/home/yx/mindone/examples/stable_diffusion_v2/configs/train_config.json', use_parallel=False, warmup_steps=1000, weight_decay=0.01)
    random seed:  3407
    Filter small images, filter size: 256
    The first image path is /home/yx/datasets/diffusion/pokemon/img1.jpg, and the caption is a cartoon character with a potted plant on his head
    total data num: 3
    rank id 0, sample num is 1
    LatentDiffusion: Running in eps-prediction mode
    Attention: output_channels=1280, num_heads=8, dim_head=-1
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    Using tokenizer `WordPieceTokenizer` for zh.
    start loading pretrained_ckpt models/wukong-huahua-ms.ckpt
    param not load: (['first_stage_model.encoder.down.3.downsample.conv.weight', 'first_stage_model.encoder.down.3.downsample.conv.bias', 'first_stage_model.decoder.up.0.upsample.conv.weight', 'first_stage_model.decoder.up.0.upsample.conv.bias'], [])
    end loading ckpt
    start_training...
    epoch: 1 step: 1, loss is 0.084607236
    Train epoch time: 356923.002 ms, per step time: 356923.002 ms
    epoch: 2 step: 1, loss is 0.05656011
    Train epoch time: 1029.360 ms, per step time: 1029.360 ms
    epoch: 3 step: 1, loss is 0.118593365
    Train epoch time: 1001.919 ms, per step time: 1001.919 ms
    epoch: 4 step: 1, loss is 0.017298669
    Train epoch time: 1006.227 ms, per step time: 1006.227 ms
    epoch: 5 step: 1, loss is 0.109166086
    Train epoch time: 1005.063 ms, per step time: 1005.063 ms
    epoch: 6 step: 1, loss is 0.097116366
    Train epoch time: 1006.625 ms, per step time: 1006.625 ms
    epoch: 7 step: 1, loss is 0.10514955
    Train epoch time: 1004.190 ms, per step time: 1004.190 ms
    epoch: 8 step: 1, loss is 0.13900207
    Train epoch time: 1003.557 ms, per step time: 1003.557 ms
    epoch: 9 step: 1, loss is 0.2265796
    Train epoch time: 1003.004 ms, per step time: 1003.004 ms
    epoch: 10 step: 1, loss is 0.0870845
    Train epoch time: 1003.814 ms, per step time: 1003.814 ms
    epoch: 11 step: 1, loss is 0.06785656
    Train epoch time: 1000.422 ms, per step time: 1000.422 ms
    epoch: 12 step: 1, loss is 0.025875507
    Train epoch time: 873.413 ms, per step time: 873.413 ms
    epoch: 13 step: 1, loss is 0.07849076
    Train epoch time: 864.334 ms, per step time: 864.334 ms
    epoch: 14 step: 1, loss is 0.07433228
    Train epoch time: 858.342 ms, per step time: 858.342 ms
    epoch: 15 step: 1, loss is 0.030560175
    Train epoch time: 858.369 ms, per step time: 858.369 ms
    epoch: 16 step: 1, loss is 0.053589523
    Train epoch time: 855.278 ms, per step time: 855.278 ms
    epoch: 17 step: 1, loss is 0.06647466
    Train epoch time: 860.348 ms, per step time: 860.348 ms
    epoch: 18 step: 1, loss is 0.03060301
    Train epoch time: 854.233 ms, per step time: 854.233 ms
    epoch: 19 step: 1, loss is 0.051647402
    Train epoch time: 853.788 ms, per step time: 853.788 ms
    epoch: 20 step: 1, loss is 0.094533354
    Train epoch time: 863.531 ms, per step time: 863.531 ms

