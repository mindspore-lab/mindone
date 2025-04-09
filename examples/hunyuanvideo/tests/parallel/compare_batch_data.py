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
max_batches_to_save = 5
for i_batch in range(max_batches_to_save):
    file_path1 = f"./rank0_batch{i_batch}.pkl"
    file_path2 = f"./rank1_batch{i_batch}.pkl"
    print(f"Comparing rank0 and rank1 from batch {i_batch}")
    compare_pkl_files(file_path1, file_path2)

    if i_batch > 0:
        print(f"Comparing batch{i_batch} and batch{i_batch-1} from rank0")
        compare_pkl_files(f"./rank0_batch{i_batch}.pkl", f"./rank0_batch{i_batch-1}.pkl")
