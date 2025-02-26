## Quick Start

### Requirement

```
mindone==0.2.0
mindspore>=2.4.0
numpy>=2.1.3
transformers>=4.38.2
```

Our development and verification are based on:
| mindspore  | ascend driver  |  firmware   | cann toolkit/kernel |
|:----------:|:--------------:|:-----------:|:------------------:|
|   2.4.0    |    24.1.RC2    | 7.3.0.1.231 |   8.0.RC3.beta1    |


### Prepare and usage

Install driver, firmware, cann toolkit and kernel with [run package](https://www.hiascend.com/developer/download/community/result?module=cann) that match your device.


Install required modules
```
pip install mindspore==2.4.0 transformers==4.38.2 numpy==2.1.3
```

Install mindone with source
```sh
git clone https://github.com/mindspore-lab/mindone.git

cd mindone

pip install -e .

cd ./examples/dit_infer_acceleration/stable_diffusion_3/eval
```

Download weights
```sh
git lfs install

git clone https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
```

Run the script
```sh
python3 sd3_infer.py --prompt "A car with flying wings" --cache_and_prompt_gate --use_graph_mode --ckpt "./stable-diffusion-3-medium-diffusers"
```
Options:
- `--ckpt`: path to Stable Diffusion 3 pretrained checkpoint
- `--prompt`: prompt or prompt list for batch images
- `--negative_prompt`: negative prompt, the quantity should correspond to the arg "prompt"
- `--cache_and_prompt_gate`: apply DiTCache and PromptGate algorithms
- `--todo`: apply ToDo algorithm
- `--max_seq_len`: maxmium input prompt length
- `--image_size`: output image size, default value is [1024, 1024](height, width)
- `--save_path`: output image save path
- `--use_graph_mode`: use graph mode to accelerate, there will be **warm up** if using graph mode

tips:

for faster acceleration, you can also bind cpu cores to run above commands with `taskset` command.
Get numa node by `lscpu`, result like this
```
NUMA node0: 0-23
NUMA node1: 24-47
NUMA node2: 48-71
NUMA node3: 72-95
```

use node0 to run
```sh
task -c 0-23 python3 sd3_infer.py --prompt "a piano on the lake" --cache_and_prompt_gate --todo --use_graph_mode
```
