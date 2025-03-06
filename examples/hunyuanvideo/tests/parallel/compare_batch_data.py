import pickle

import numpy as np


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def compare_arrays(arr1, arr2, rtol=1e-05, atol=1e-08):
    return np.allclose(arr1, arr2, rtol=rtol, atol=atol)


def compare_pkl_files(file_path1, file_path2):
    data1 = load_pkl(file_path1)
    data2 = load_pkl(file_path2)

    if len(data1) != len(data2):
        print(
            f"Error: The number of arrays in the files is different. File 1 has {len(data1)} arrays, File 2 has {len(data2)} arrays."
        )
        return

    for i, (arr1, arr2) in enumerate(zip(data1, data2)):
        if not compare_arrays(arr1, arr2):
            print(f"Arrays at index {i} are not close.")
        else:
            print(f"Arrays at index {i} are close.")


# Example usage
file_path1 = "./batch0_rank0.pkl"
file_path2 = "./batch0_rank1.pkl"
compare_pkl_files(file_path1, file_path2)
