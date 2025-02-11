import logging
import os
from typing import List

import numpy as np

import mindspore as ms

_logger = logging.getLogger(__name__)


class PerfRecorder(object):
    def __init__(
        self,
        save_dir,
        metric_names: List = ["step", "loss", "train_time(s)"],
        file_name="result.log",
        separator="\t",
        resume=False,
    ):
        self.save_dir = save_dir
        self.sep = separator
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            _logger.info(f"{save_dir} not exist. Created.")

        self.log_txt_fp = os.path.join(save_dir, file_name)
        if not resume:
            result_log = separator.join(metric_names)
            with open(self.log_txt_fp, "w", encoding="utf-8") as fp:
                fp.write(result_log + "\n")

    def add(self, step, *measures):
        """
        measures (Tuple): measurement values corresponding to the metric names
        """
        sep = self.sep
        line = f"{step}{sep}"
        for i, m in enumerate(measures):
            if isinstance(m, ms.Tensor):
                m = m.asnumpy()

            if isinstance(m, float) or isinstance(m, np.float32):
                line += f"{m:.4f}"
            elif m is None:
                line += "NA"
            else:
                line += f"{m:.10}"

            if i < len(measures) - 1:
                line += f"{sep}"

        with open(self.log_txt_fp, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")


if __name__ == "__main__":
    r = PerfRecorder("./")
    r.add(1, 0.2, 0.4, 0.5, 199)
