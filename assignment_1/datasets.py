import torch
from torch.utils.data import Dataset


class LaserDataset(Dataset):
    """
    Dataset for laser time series data.
    """
    def __init__(self, data, sequence_length: int, forecast_horizon=1) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1

    def __getitem__(self, idx: int):
        input_seq_end = idx + self.sequence_length
        target_idx = input_seq_end + self.forecast_horizon - 1

        X = self.data[idx:input_seq_end]
        y = self.data[target_idx]

        return X.unsqueeze(-1), y
