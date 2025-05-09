from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torch.nn import Module
from torch.optim import Optimizer


def train_model(
    dataloader: DataLoader,
    model: Module,
    optimizer: Optimizer,
    loss_fc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int
) -> None:
    model.train()
    for _ in tqdm(range(epochs)):
        for X, y in dataloader:
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fc(output, y)
            loss.backward()
            optimizer.step()


def validate_model(
    dataloader: DataLoader,
    model: Module,
    loss_fc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            output = model(X)
            loss = loss_fc(output, y)
            val_loss += loss.item()

    return val_loss / len(dataloader)


def hypertune_objective(
    model_cls: Module,
    optim_cls: Optimizer,
    dataset_cls: Dataset,
    data: np.array,
    transforms: list,
    loss_fc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int,
    k_folds: int = 5,
    batch_size: int = 32,
    model_params: Optional[Dict] = {},
    optim_params: Optional[Dict] = {},
    data_params: Optional[Dict] = {},
) -> float:
    dataset = dataset_cls(data, transforms=transforms, **data_params)

    kfold = KFold(n_splits=k_folds)
    fold_losses = []

    for _, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                  shuffle=False,)
        val_loader = DataLoader(val_subset, batch_size=batch_size,
                                shuffle=False)

        model = model_cls(input_size=data_params['window_size'],
                          **model_params)
        optimizer = optim_cls(model.parameters(), **optim_params)

        train_model(train_loader, model, optimizer, loss_fc, epochs)
        avg_val_loss = validate_model(val_loader, model, loss_fc)

        fold_losses.append(avg_val_loss)

    avg_cv_loss = np.mean(fold_losses)
    return avg_cv_loss


def generate_predictions(
    model: Module,
    data: np.array,
    sequence_length: int = 50,
    horizon: int = 200
) -> None:
    model.eval()

    data_tensor = torch.tensor(data, dtype=torch.float32)
    history = data_tensor[-sequence_length:].view(1, -1)

    generated = []
    current_seq = history.clone()
    with torch.no_grad():
        for _ in range(horizon):
            next_val = model(current_seq)
            generated.append(next_val.item())
            current_seq = torch.cat([current_seq[:, 1:], next_val], dim=1)

    plt.figure(figsize=(25, 7))
    plt.plot(range(len(data_tensor)), data_tensor.tolist(),
             label="Training Data")
    plt.plot(range(len(data_tensor), len(data_tensor) + horizon), generated,
             label="Generated", color='red')
    plt.legend()
    plt.title(f"Forecast: Training Data + {horizon}-Step Prediction")
    plt.grid(True)
    plt.show()


def generate_predictions_with_train_fit(
    model,
    data: np.array,
    sequence_length: int = 50,
    horizon: int = 200
) -> None:
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32)

    preds_train = []
    with torch.no_grad():
        for i in range(sequence_length, len(data_tensor)):
            input_seq = data_tensor[i - sequence_length:i].view(1, -1)
            pred = model(input_seq)
            preds_train.append(pred.item())

    history = data_tensor[-sequence_length:].view(1, -1)
    generated = []
    current_seq = history.clone()
    with torch.no_grad():
        for _ in range(horizon):
            next_val = model(current_seq)
            generated.append(next_val.item())
            current_seq = torch.cat([current_seq[:, 1:], next_val], dim=1)

    plt.figure(figsize=(25, 7))
    plt.plot(range(len(data_tensor)), data_tensor.tolist(), label="Training Data")
    plt.plot(range(sequence_length, len(data_tensor)), preds_train, label="Train Fit", color='red')
    plt.plot(range(len(data_tensor), len(data_tensor) + horizon), generated, label="Forecast", color='red')
    plt.legend()
    plt.title(f"Model Fit + {horizon}-Step Forecast")
    plt.grid(True)
    plt.show()


def final_model_evaluation(
    model,
    test_data: np.array,
    train_data: np.array,
    scaler,
    input_size: int = 50,
    horizon: int = 200
) -> None:
    model.eval()

    seed_tensor = torch.tensor(train_data[-input_size:], dtype=torch.float32).view(1, -1)

    generated = []
    current_seq = seed_tensor.clone()
    with torch.no_grad():
        for _ in range(horizon):
            next_val = model(current_seq)
            generated.append(next_val.item())
            current_seq = torch.cat([current_seq[:, 1:], next_val], dim=1)

    predictions = scaler.inverse_transform(np.array(generated).reshape(-1, 1))
    test_data = scaler.inverse_transform(test_data[:horizon])

    mse = mean_squared_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)

    plt.figure(figsize=(25, 7))
    plt.plot(range(horizon), test_data[:horizon].flatten(), label="Test Data")
    plt.plot(range(horizon), predictions, label="Predicted", color='red')
    plt.legend()
    plt.title(f"Prediction vs Test Data| MSE: {mse:.4f}, MAE: {mae:.4f}")
    plt.grid(True)
    plt.show()
