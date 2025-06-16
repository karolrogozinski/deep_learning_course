import os

import numpy as np
from torch.utils.data import Dataset

from utils import get_data_matrix


class MEGDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            is_train: bool = True,
            downsampling_size: int = 248
            ):
        self.folder_path = data_path
        self.file_paths = sorted([
            os.path.join(data_path, fname)
            for fname in os.listdir(data_path)
        ])
        self.num_samples = downsampling_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx) -> np.array:
        file_path = self.file_paths[idx]
        data, name = get_data_matrix(file_path)

        if self.is_train:
            data = self._random_downsample_data(data)
            data = self._normalize_data(data)

        data = data.astype(np.float32)
        label = self._get_label(name)

        return data, label

    def _random_downsample_data(self, data: np.array) -> np.array:
        length = data.shape[0]
        indices = np.sort(
            np.random.choice(length, self.num_samples, replace=False)
            )
        return data[:, indices]

    def _normalize_data(self, data: np.array) -> np.array:
        data = (data - data.min()) / (data.max() - data.min())
        return data
    
    def _get_label(self, name: str) -> str:
        clear_name = ''.join(name.split('_')[:-1])

        dict_label = {
            'rest': 0,
            'taskmotor': 1,
            'taskstorymath': 2,
            'taskworkingmemory': 3,
        }

        label = dict_label[clear_name]
        return label
