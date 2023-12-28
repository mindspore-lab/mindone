import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _generate_colormap(num_of_labels: int) -> np.ndarray:
    """
    Generates a colormap based on the number of labels in a dataset.
    The mapping function is based on the
    `PITI <https://github.com/PITI-Synthesis/PITI/blob/main/preprocess/preprocess_mask.py>`__ project.

    Parameters:
        num_of_labels: The number of labels in a dataset.

    Returns:
        A numpy array of shape (256, 3) containing the RGB values for each label id.
    """
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(num_of_labels):
        r, g, b = 0, 0, 0
        _id = i + 1
        for j in range(7):
            str_id = f"{_id:03b}"
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            _id //= 8
        cmap[i] = [r, g, b]

    return cmap


def colorize_masks(in_dir: Path, out_dir: Path):
    """
    Colorizes COCO-Stuff masks in a given input directory and saves the colored masks in an output directory.

    Args:
        in_dir: The input directory containing the grayscale COCO-Stuff masks.
        out_dir: The output directory where the RGB colored masks will be saved.
    """
    colormap = _generate_colormap(182)  # 182 classes in the COCO-Stuff dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    def color_mask(mask_path):
        colored_mask_path = out_dir / mask_path.name
        image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        image = colormap[image]
        cv2.imwrite(str(colored_mask_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    mask_paths = list(in_dir.iterdir())
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(color_mask, mask_paths), total=len(mask_paths)))


if __name__ == "__main__":
    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    colorize_masks(in_path, out_path)
