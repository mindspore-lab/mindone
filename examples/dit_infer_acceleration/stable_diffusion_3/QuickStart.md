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

### Usage

Run eval script fellow
```
cd <path_to_mindone>/examples/dit_infer_acceleration/stable_diffusion_3/eval

python3 eval_sd3_infer.py --prompt "A car with flying wings" --cache_and_prompt_gate --use_graph_mode
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
