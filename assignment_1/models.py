import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ExtendedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, dropout_prob=0.5):
        super(ExtendedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.lstm(x)
        return self.fc(h[-1])


class TimeSeriesCNN(nn.Module):
    def __init__(self, input_size, n_filters=64, kernel_size=3):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((input_size - 2*(kernel_size-1)) * n_filters, 1)

    def forward(self, x):
        # x: (batch, time_steps)
        x = x.unsqueeze(1)  # (batch, 1, time_steps)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)
