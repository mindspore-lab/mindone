from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple


class BaseDataset(ABC):
    """
    An abstract base class for datasets in MindONE package.

    Required attributes:
        output_columns: A list of column names that a dataset is expected to output.
        pad_info: When it is not None, that dataloader will call `padded_batch` instead of `pad` in the final step, and
            `pad_info` will be passed as an argument in `padded_batch` method. For detail usage, please check
            https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/batch/mindspore.dataset.Dataset.padded_batch.html
    """

    output_columns: List[str]
    pad_info: Optional[Dict[str, Tuple[Sequence[int], int]]]

    @abstractmethod
    def __getitem__(self, idx):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @staticmethod
    @abstractmethod
    def train_transforms(**kwargs) -> List[dict]:
        """
        Defines a list transformations that should be applied to the training data in the following form:

        {
            "operations": [List of transform operations],               # Required
            "input_columns": [List of columns to apply transforms to],  # Optional
            "output_columns": [List of output columns]                  # Optional, only used if different from the `input columns`
        }

        Args:
            **kwargs: additional parameters to be passed to the transformations.

        Returns:
            A list of transformations that should be applied to the training data.

        Examples:
            >>> import numpy as np
            >>> from mindspore.dataset.vision import Resize, ToTensor
            >>>
            >>> def train_transforms(tokenizer) -> List[dict]:
            >>>     transforms = [
            >>>         {
            >>>             "operations": tokenizer,
            >>>             "input_columns": ["caption"],
            >>>         },
            >>>         {
            >>>             "operations": [Resize((512, 512)), lambda x: (x / 127.5 - 1.0).astype(np.float32)],
            >>>             "input_columns": ["image"],
            >>>         },
            >>>         {
            >>>             "operations": [Resize((512, 512)), ToTensor()],
            >>>             "input_columns": ["condition"],
            >>>         },
            >>>     ]
            >>>
            >>>     return transforms
        """
        ...
