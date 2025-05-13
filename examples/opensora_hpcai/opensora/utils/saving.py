import os
from dataclasses import dataclass
from datetime import datetime

from jsonargparse.typing import path_type

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


@dataclass
class SavingOptions:
    output_path: Path_dcc = os.path.join(os.getcwd(), "samples", datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    fps: int = 24

    def __post_init__(self):
        self.output_path = os.path.abspath(self.output_path)


@dataclass
class TrainingSavingOptions:
    output_path: Path_dcc = os.path.join(os.getcwd(), "output", datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))

    def __post_init__(self):
        self.output_path = os.path.abspath(self.output_path)
