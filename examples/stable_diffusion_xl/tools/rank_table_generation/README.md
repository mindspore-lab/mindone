# Generate Rank Table File

MindSpore distributed training launch helper utility that will generate hccl config file.

## 1. Generate rank_table file with a single server

This script generates a rank_table_file for a single server by using `hccn_tool` or read `/etc/hccn.conf`.

### Usage

```bash
# generate rank_table file with 8p
python hccl_tools.py

# generate rank_table file with others
python hccl_tools.py --device_num "[0,4)"
```

### Note

Please note that the Ascend accelerators used must be continuous, such [0,4) means to use four chips 0，1，2，3; [0,1) means to use chip 0; The first four chips are a group, and the last four chips are a group. In addition to the [0,8) chips are allowed, other cross-group such as [3,6) are prohibited.

`--visible_devices` means the visible devices according to the software system. Usually used in the virtual system or docker container that makes the device_id dismatch logic_id. `--device_num` uses logic_id. For example "4,5,6,7" means the system has 4 logic chips which are the last 4 chips in hardware while `--device_num` could only be set to "[0, 4)" instead of "[4, 8)"

`hccl_tools` used `/etc/hccn.conf` to generate rank_table_file. `/etc/hccn.conf` is the configuration file about ascend accelerator resources.


## 2. Merge rank_table files from multi-server

This script is used to merge server rank_table_file for a single server into one file for the cluster.

### Usage

```bash
# merge hccl1.json and hccl2.json
python merge_hccl.py hccl1.json hccl2.json

# merge all hccl*.json
python merge_hccl.py hccl*.json

# merge with a specific save path for merged file
python merge_hccl.py hccl1.json hccl2.json --save_path specific/save/path
```

Example uasge of distributed vallina training using rank table file.

```shell
# sdxl-base fine-tune with 8p
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_8p.json 0 8 8 /path_to/dataset/

# sdxl-base fine-tune with 16p on Ascend
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_16p.json 0 8 16 /path_to/dataset/  # run on server 1
bash scripts/run_distribute_vanilla_ft_910b.sh /path_to/hccl_16p.json 8 16 16 /path_to/dataset/ # run on server 2
```

### Note

The server order in the output config file comes from the order of the input file list.

For example, running `python merge_hccl.py hccl_1.json hccl_2.json`. The 8 devices in hccl_1.json will be ranked `0~7`, and the 8 devices in hccl_2.json will be ranked `8~15`.

While running with a wildcard, the exact order is not determined, which is decided by the system. Usually, this will result in dictionary order just like `ls` command, but we still suggest you check the result carefully if the order does matter in your situation.
