import errno
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch
from tqdm import tqdm

from mindone.metrics.utils.convert_inception_weights import transfer_torch_inception_weights


def download_url_to_file(url: str, dst: str, progress: bool = True) -> None:
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    dst = os.path.expanduser(dst)
    for seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")

    try:
        with tqdm(total=file_size, disable=not progress, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))

        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def load_state_dict_from_url(
    url: str, save_dir: Optional[str] = None, progress: bool = True, file_name: Optional[str] = None
) -> Dict[str, Any]:
    if save_dir is None:
        default_dir = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(default_dir, "checkpoints")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(save_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, progress)

    state_dict = torch.load(cached_file)

    return transfer_torch_inception_weights(state_dict)
