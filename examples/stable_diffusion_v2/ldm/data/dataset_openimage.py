import mindspore as ms
import numpy as np
import mindspore.dataset.vision as vision
from tqdm import tqdm
import csv
from multiprocessing import cpu_count

def crop(x, size=(512, 512)):
    x = vision.RandomCrop(size=size, pad_if_needed=True)(x)
    return x


class DistributedSampler:
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xFFFFFFFF
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len)
        else:
            indices = np.arange(self.dataset_len)
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.dataset_len


class OpenImage():
    def __init__(self, filename, folder):
        self.bins = []
        with open(filename, newline='\n') as f:
            reader = csv.DictReader(f)
            for line in tqdm(reader):
                self.bins.append(f'{folder}/%s.jpg' % line['ImageID'])
        np.random.seed(0)
        np.random.shuffle(self.bins)

    def __getitem__(self, index):
        return self.bins[index]

    def __len__(self):
        return len(self.bins)


def create_openimage_dataset(
    metadata='train',
    data_folder='openimage_test',
    batch_size=1,
    use_fp16=False,
    rank=0,
    group_size=1,
    *args,
    **kwargs
):
    w = kwargs.get('resolution', 256)

    np.random.seed(0)
    ds = OpenImage(metadata, data_folder)
    input_columns = ['image']
    sampler = DistributedSampler(ds, rank, group_size, shuffle=True)
    ds = ms.dataset.GeneratorDataset(
        ds,
        column_names=input_columns,
        sampler=sampler
    )
    def read_feat(path):
        path = str(path).replace('b\'', '').replace('\'', '')
        with open(path, 'rb') as f:
            img = vision.Decode()(f.read())
            x = crop(img, (w, w)).transpose([2, 0, 1])
            x = x / 255.
            x = x * 2. - 1.
            return x.astype(np.float16 if use_fp16 else np.float32)

    ds = ds.map(
        input_columns=['image'],
        output_columns=input_columns,
        column_order=input_columns,
        operations=[read_feat],
        num_parallel_workers=cpu_count(),
    )

    ds = ds.batch(
        batch_size, 
        drop_remainder=True,
        python_multiprocessing=False,
        num_parallel_workers=cpu_count()
    )

    return ds
