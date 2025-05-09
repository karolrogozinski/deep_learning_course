import torch
from torch.utils.data import Dataset


class LaserDataset(Dataset):
    """
    Dataset for laser time series data.
    """
    def __init__(self, data, window_size: int, forecast_horizon=1, transforms=None) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = window_size
        self.forecast_horizon = forecast_horizon
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1

    def __getitem__(self, idx: int):
        input_seq_end = idx + self.sequence_length
        target_idx = input_seq_end + self.forecast_horizon - 1

        X = self.data[idx:input_seq_end].view(-1)
        y = self.data[target_idx]

        for transform in self.transforms:
            X = transform(X)

        return X, y
