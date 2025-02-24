

## installation

```shell
pip install git+https://github.com/zhanghuiyao/mindone.git@step_video
```


## todo list

- speed-up
- ...



## inference


### step 1: (option but recommend) download weights

link: https://huggingface.co/stepfun-ai/stepvideo-t2v



### step 2: convert a `.bin` weight (hunyuan-clip) format from `pytorch_model.bin` to `model.safetensors`

```shell
python convert.py --pt_filename where_bin_file --sf_filename where_safetensors_file --config_path where_{config.json}_file

# example as:
python convert.py --pt_filename /path_to/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin --sf_filename /path_to/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/model.safetensors --config_path /path_to/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/config.json
```


### step 3: running

```shell
model_dir=where_you_download_dir

# (1) start vae/captioner server on single-card (Ascend910*)
# !!! This command will return the URL for both the caption API and the VAE API. Please use the returned URL in the following command.
ASCEND_RT_VISIBLE_DEVICES=0 python api/call_remote_server.py --model_dir $model_dir --enable_vae True &
ASCEND_RT_VISIBLE_DEVICES=1 python api/call_remote_server.py --model_dir $model_dir --enable_llm True &



# !!! wait...a moment, vae/llm is loading...



# (2) setting and replace the `url` from before command print
parallel=4
sp=2
pp=2
vae_url='127.0.0.1'
caption_url='127.0.0.1'


# (3) run parallel dit model on 4-cards (Ascend910*)
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 msrun --bind_core=True --worker_num=$parallel --local_worker_num=$parallel --master_port=9000 --log_dir=outputs/parallel_logs python -u \
run_parallel.py --model_dir $model_dir --vae_url $vae_url --caption_url $caption_url  --ulysses_degree $sp --pp_degree $pp --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 30  --cfg_scale 9.0 --time_shift 13.0 --num_frames 136 --height 544 --width 992
```


## performence

|     Model    |  height/width/frame |  Peak Memory | 50 steps |
|:------------:|:------------:|:------------:|:------------:|
| Step-Video-T2V   |        544px992px204f      |  45.83 GB | ~52 min |
| Step-Video-T2V   |        544px992px136f      |  40.48 GB | ~35 min |



<br>
<br>
<br>
<br>
<br>


## test scripts


### 1. test mini-step-video on single-process and single-card

```shell
parallel=1
url='127.0.0.1'
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

python run_single_test.py --model_dir $model_dir --vae_url $url --caption_url $url  --ulysses_degree $parallel --pp_degree 1 --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 5  --cfg_scale 9.0 --time_shift 13.0 --num_frames 16 --height 128 --width 128
```


```shell
parallel=4
sp=2
pp=2
url='127.0.0.1'
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 msrun --bind_core=True --worker_num=$parallel --local_worker_num=$parallel --master_port=9000 --log_dir=outputs/parallel_logs python \
run_single_test.py --model_dir $model_dir --vae_url $url --caption_url $url  --ulysses_degree $sp --pp_degree $pp --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 5  --cfg_scale 9.0 --time_shift 13.0 --num_frames 16 --height 128 --width 128

tail -f outputs/parallel_logs/worker_0.log
```


#### 1.1. test vae decode

```shell
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

python test_vae_decode.py --model_dir $model_dir --ulysses_degree 1 --pp_degree 1
```


#### 1.1.1. test llm

```shell
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

python test_llm.py --model_dir $model_dir --ulysses_degree 1 --pp_degree 1 --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光"
```



#### 1.2. test dit


```shell
parallel=1
url='127.0.0.1'
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

ASCEND_RT_VISIBLE_DEVICES=5 python -u test_dit_infer.py --model_dir $model_dir --vae_url $url --caption_url $url  --ulysses_degree 1 --pp_degree 1 --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 5  --cfg_scale 9.0 --time_shift 13.0 --num_frames 16 --height 128 --width 128
```



```shell
parallel=4
sp=2
pp=2
url='127.0.0.1'
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 msrun --bind_core=True --worker_num=$parallel --local_worker_num=$parallel --master_port=9000 --log_dir=outputs/parallel_logs python \
test_dit_infer.py --model_dir $model_dir --vae_url $url --caption_url $url  --ulysses_degree $sp --pp_degree $pp --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 5  --cfg_scale 9.0 --time_shift 13.0 --num_frames 16 --height 128 --width 128

tail -f outputs/parallel_logs/worker_0.log
```

#### 1.2. test dit sp+pp

```shell
parallel=4
sp=2
pp=2
url='127.0.0.1'
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 msrun --bind_core=True --worker_num=$parallel --local_worker_num=$parallel --master_port=9000 --log_dir=outputs/parallel_logs python \
test_dit_infer.py --model_dir $model_dir --vae_url $url --caption_url $url  --ulysses_degree $sp --pp_degree $pp --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 5  --cfg_scale 9.0 --time_shift 13.0 --num_frames 136

tail -f outputs/parallel_logs/worker_0.log
```


#### 1.3. test send/recv

```shell
parallel=4

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 msrun --bind_core=True --worker_num=$parallel --local_worker_num=$parallel --master_port=9000 --log_dir=outputs/parallel_logs python test_send_recv.py
```


### 2. test server connect

```shell
url='127.0.0.1'
model_dir='./demo/stepfun-ai/stepvideo-t2v_mini'

# run vae/captioner server on single-card (Ascend910*)
ASCEND_RT_VISIBLE_DEVICES=5 python api/call_remote_server.py --model_dir $model_dir --enable_vae True &
ASCEND_RT_VISIBLE_DEVICES=6 python api/call_remote_server.py --model_dir $model_dir --enable_llm True &

# run main process on single-cards (Ascend910*)
ASCEND_RT_VISIBLE_DEVICES=7 python run_server_test.py --model_dir $model_dir --vae_url $url --caption_url $url  --ulysses_degree 1 --pp_degree 1 --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 5  --cfg_scale 9.0 --time_shift 13.0 --num_frames 16
```



### 3. test full-size memory and loading checkpoint
 
 ...
