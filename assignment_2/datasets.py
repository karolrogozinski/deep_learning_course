import os

import numpy as np

import torch
from torch.utils.data import Dataset

from utils import get_data_matrix


class MEGDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            mode: str = 'train',
            transforms: tuple = None,
            downsampling_size: int = 248,
            continous: bool = False,
            multiple: bool = False,
            num_segments: int = 7,
            val_samples: tuple[str] = ('7', '8'),
            person_id: bool = False,
            per_person_norm: dict[dict] = None,
            ):
        self.folder_path = data_path
        self.num_samples = downsampling_size
        self.mode = mode
        self.continous = continous
        self.multiple = multiple
        self.num_segments = num_segments
        self.per_person_norm = per_person_norm
        self.transforms = transforms
        self.person_id = person_id
        self._load_file_paths(data_path, val_samples)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx) -> np.array:
        file_path = self.file_paths[idx]
        data, name = get_data_matrix(file_path)
        person = self._get_person(name)

        data = data * 1e10

        if self.per_person_norm is not None:
            data = self._normalize_per_person(data, person)
        else:
            data = self._normalize_data(data)

        if self.continous:
            if self.multiple:
                data = self._continuous_downsample_multiple(data)
            else:
                data = self._continuous_downsample_data(data)
        else:
            data = self._random_downsample_data(data)

        data = torch.tensor(data, dtype=torch.float32)
        if not self.multiple:
            data = data.unsqueeze(0)
        label = self._get_label(name)

        for transform in self.transforms:
            data = transform(data)

        if self.person_id:
            return data, (label, self._get_person_id(person))
        return data, label

    def _random_downsample_data(self, data: np.array) -> np.array:
        length = data.shape[1]
        indices = np.sort(
            np.random.choice(length, self.num_samples, replace=False)
            )
        return data[:, indices]

    def _continuous_downsample_data(self, data: np.array) -> np.array:
        length = data.shape[1]
        max_start = length - self.num_samples

        start = np.random.randint(0, max_start + 1)
        end = start + self.num_samples

        return data[:, start:end]

    def _continuous_downsample_multiple(self, data: np.array) -> np.array:
        length = data.shape[1]
        max_start = length - self.num_samples
        segments = []

        for _ in range(self.num_segments):
            start = np.random.randint(0, max_start + 1)
            end = start + self.num_samples
            segment = data[:, start:end]
            segments.append(segment)

        return np.stack(segments)

    def _normalize_data(self, data: np.array) -> np.array:
        # means = data.mean(axis=1, keepdims=True)
        # stds = data.std(axis=1, keepdims=True)
        # eps = 1e-8

        # normalized = (data - means) / (stds + eps)
        normalized = (data - data.mean()) / (data.std())

        return normalized

    def _normalize_per_person(self, data: np.array, person: str) -> str:
        data = (
            data - self.per_person_norm[person]["mean"][:, None]
        ) / (
            self.per_person_norm[person]["std"][:, None] + 1e-8
        )
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

    def _get_person(self, name: str) -> str:
        person = name.split('_')[-1]
        return person

    def _get_person_id(self, person: str) -> str:
        dict_person = {
            '105923': 0,
            '113922': 0,
            '164636': 1,
            '725751': 0,
            '735148': 0,
            '707749': 0,
            '162935': 0,
        }
        return dict_person[person]

    def _load_file_paths(self, data_path: str, val_samples: tuple[str]) -> None:
        if self.mode == 'train':
            self.file_paths = sorted([
                os.path.join(data_path, fname)
                for fname in os.listdir(data_path)
                if fname[-4] not in val_samples
            ])
        elif self.mode == 'valid':
            self.file_paths = sorted([
                os.path.join(data_path, fname)
                for fname in os.listdir(data_path)
                if fname[-4] in val_samples
            ])
        elif self.mode == 'test':
            self.file_paths = sorted([
                os.path.join(data_path, fname)
                for fname in os.listdir(data_path)
            ])
        else:
            raise ValueError(f'{self.mode}: given mode not exist')
