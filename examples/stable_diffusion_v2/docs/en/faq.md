# FAQ - Frequently Asked Questions

### 1. Error reported when using `ms.SummaryCollector` to collect data.

<details>

#### [issue 522](https://github.com/mindspore-lab/mindone/issues/522): [ERROR][mindspore/train/summary/_summary_adapter.py:363] The dimension of Summary tensor should be 4 or second dimension should be 1 or 3, but got tag = input_data/auto, ndim = 4, shape=(2, 512, 512, 3), which means Summary tensor is not Image.

### Answer:

#### (1) Due to the requirement of `collect_input_data`, the input data format for the second dimension is channels, while the input data format for sdv2 is (batch_size, H, W, channels), and the fourth dimension is channels, the above error will be reported.

#### (2) Currently not supported for `collect_input_data` to collect data. Please set `collect_input_data` is False for `ms.SummaryCollector`. For example:
```shell
   interval_1 = [x for x in range(1, 4)]
   specified = {"collect_metric": True, "histogram_regular": "^conv1.|^conv2.", "collect_graph": True,
    "collect_dataset_graph": True,"collect_train_lineage":True,"collect_input_data":False,
    'collect_landscape': {'landscape_size': 40,'unit': "epoch",'create_landscape': {'train': True,'result': False},
    'num_samples': 32,'intervals': [interval_1]}
    }
   summary_collector = ms.SummaryCollector(summary_dir="./summary_dir/summary_01", collect_specified_data=specified,
    collect_freq=12, keep_default_action=False)
   callback.append(summary_collector)
  ```

</details>
