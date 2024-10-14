# from torch.utils.data import ConcatDataset, Dataset
import mindspore.dataset

from examples.nextgpt.core_code.dataset.catalog import DatasetCatalog
from examples.nextgpt.core_code.dataset.utils import instantiate_from_config
# from mindspore import dataset

class MyConcatDataset():
    def __init__(self, dataset_name_list):
        super(MyConcatDataset, self).__init__()

        _datasets = []

        catalog = DatasetCatalog()
        for dataset_idx, dataset_name in enumerate(dataset_name_list):
            dataset_dict = getattr(catalog, dataset_name)

            target = dataset_dict['target']
            params = dataset_dict['params']
            print(target)
            print(params)
            dataset_ = instantiate_from_config(dict(target=target, params=params))
            _datasets.append(dataset_)
        # self.datasets = _datasets
        self.datasets = _datasets
    def __len__(self):
        return self.datasets.__len__()

    def __getitem__(self, item):
        return self.datasets.__getitem__(item)

    def collate(self, instances):
        data = {key: [] for key in instances[0].keys()} if instances else {}

        for instance in instances:
            for key, value in instance.items():
                data[key].append(value)

        return data

if __name__ == '__main__':
    dataset_name_list = ['cc3m_enc','webvid_enc']
    concat_data = MyConcatDataset(dataset_name_list)
    sample = mindspore.dataset.RandomSampler()
    for data_list in concat_data:
        data_loader = mindspore.dataset.GeneratorDataset(data_list,['data_list'])
        print(data_loader)
        for data in data_loader:
            print(data)
            print("--------")