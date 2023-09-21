from multiprocessing import cpu_count

import numpy as np
from audioio import read
from data.tangopromptbank import TangoPromptBank

import mindspore as ms
import mindspore.dataset.audio as msaudio

data_columns = [
    "audio",
    "text",
]


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
        indices = indices[self.rank :: self.group_size]
        return iter(indices)

    def __len__(self):
        return self.dataset_len


def build_dataset(args, group_size, rank, tokenizer, is_train=True):
    ds = TangoPromptBank(
        data_path=args.data_path,
        is_train=is_train,
    )
    input_columns = ["audio_path", "text"]
    sampler = DistributedSampler(ds, rank, group_size, shuffle=True)
    ds = ms.dataset.GeneratorDataset(ds, column_names=input_columns, sampler=sampler)

    def read_wav(filename, sr=16000, target_length=160000):
        filename = str(filename).replace("b'", "").replace("'", "")
        _wav, _sr = read(filename)
        if len(_wav.shape) > 1:
            _wav = _wav.mean(-1)
        if _sr != sr:
            _wav = msaudio.Resample(orig_freq=_sr, new_freq=sr)(_wav)
        # normalize
        _wav = _wav - np.mean(_wav)
        _wav = _wav / (np.max(np.abs(_wav)) + 1e-8)
        _wav = _wav * 0.5
        # pad
        wav = np.zeros(target_length, dtype=np.float32)
        wav[: min(len(_wav), target_length)] = _wav[:target_length]
        # ???
        wav = wav / np.max(np.abs(wav))
        wav = 0.5 * wav
        return wav

    ds = ds.map(
        input_columns=["audio_path"],
        output_columns=data_columns,
        column_order=data_columns,
        operations=[read_wav],
        num_parallel_workers=cpu_count(),
    )

    def tokenise(text):
        padding = truncation = True
        input_ids, attention_mask = tokenizer(text, tokenizer.model_max_length, padding, truncation)
        return input_ids, attention_mask

    ds = ds.map(
        input_columns=["text"],
        output_columns=["audio", "input_ids", "attention_mask"],
        column_order=["audio", "input_ids", "attention_mask"],
        operations=[tokenise],
        num_parallel_workers=cpu_count(),
    )

    ds = ds.batch(
        args.train_batch_size,
        # per_batch_map=batch_collate,
        # input_columns=feature_columns,
        # output_columns=dc,
        # column_order=dc,
        drop_remainder=True,
        python_multiprocessing=False,
        num_parallel_workers=cpu_count(),
    )

    return ds
