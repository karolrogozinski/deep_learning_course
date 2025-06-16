import h5py

import numpy as np


def get_dataset_name(filename_with_dir: str) -> str:
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name


def get_data_matrix(filename_path: str) -> np.array:
    with h5py.File(filename_path, 'r') as f:
        dataset_name = get_dataset_name(filename_path)
        matrix = f.get(dataset_name)[()]
    return matrix, dataset_name
