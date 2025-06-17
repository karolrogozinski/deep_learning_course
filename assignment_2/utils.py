from typing import Callable, Optional, Dict

import h5py
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from torch.optim import Optimizer

from tqdm import tqdm


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


def train_model(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    model: Module,
    optimizer: Optimizer,
    loss_fc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int
) -> None:
    train_losses = list()
    train_accs = list()
    valid_losses = list()
    valid_accs = list()

    model.train()
    for _ in tqdm(range(epochs)):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fc(output, y)
            loss.backward()
            optimizer.step()

        train_metrics = validate_model(train_dataloader, model, loss_fc)
        valid_metrics = validate_model(valid_dataloader, model, loss_fc)

        train_losses.append(train_metrics[0])
        train_accs.append(train_metrics[1])

        valid_losses.append(valid_metrics[0])
        valid_accs.append(valid_metrics[1])

    return train_losses, train_accs, valid_losses, valid_accs


def validate_model(
    dataloader: DataLoader,
    model: Module,
    loss_fc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> tuple[float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(next(model.parameters()).device, dtype=torch.float32)
            y = y.to(next(model.parameters()).device, dtype=torch.long)

            output = model(X)
            loss = loss_fc(output, y)
            val_loss += loss.item()

            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = val_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    model.train()

    return avg_loss, accuracy


def cross_val_model(
        data_path: str,
        model_cls: Module,
        optim_cls: Module,
        dataset_cls: Dataset,
        loss_fc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        epochs: int = 50,
        model_params: Optional[Dict] = {},
        optim_params: Optional[Dict] = {},
        data_params: Optional[Dict] = {},
) -> dict:
    splits = (('1', '2'), ('3', '4'), ('5', '6'), ('7', '8'))
    all_metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for split in splits:
        train_dataset = dataset_cls(
            data_path, mode='train', val_samples=split, **data_params)
        valid_dataset = dataset_cls(
            data_path, mode='valid', val_samples=split, **data_params)

        train_dataloader = DataLoader(
            train_dataset, batch_size=len(train_dataset), drop_last=False)
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=len(valid_dataset), drop_last=False)

        model = model_cls(**model_params)
        optimizer = optim_cls(model.parameters(), **optim_params)

        train_metrics = train_model(
            train_dataloader, valid_dataloader, model,
            optimizer, loss_fc, epochs)

        train_loss, train_acc, val_loss, val_acc = train_metrics
        all_metrics['train_loss'].append(train_loss)
        all_metrics['train_acc'].append(train_acc)
        all_metrics['val_loss'].append(val_loss)
        all_metrics['val_acc'].append(val_acc)

        print(f'Split: {split}, train_loss: {train_loss[-1]:.4f}, val_loss: {val_loss[-1]:.4f}, ' +
              f'train_acc: {train_acc[-1]:.4f}, val_acc: {val_acc[-1]:.4f}')

    result = {}
    for key in all_metrics:
        values = np.array(all_metrics[key])
        result[f'{key}_mean'] = values.mean(axis=0)
        result[f'{key}_std'] = values.std(axis=0)

    return result


def plot_cross_val_metrics(metrics: dict):
    epochs = range(1, len(metrics['train_loss_mean']) + 1)

    train_color = '#1e81b0'
    val_color = '#873e23'
    line_width = 3

    def style_axes(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Cross-Val Loss")
    ax1.plot(epochs, metrics['train_loss_mean'], label='Train Loss', color=train_color, linewidth=line_width)
    ax1.fill_between(epochs,
                     metrics['train_loss_mean'] - metrics['train_loss_std'],
                     metrics['train_loss_mean'] + metrics['train_loss_std'],
                     alpha=0.2, color=train_color)
    ax1.plot(epochs, metrics['val_loss_mean'], label='Val Loss', color=val_color, linewidth=line_width)
    ax1.fill_between(epochs,
                     metrics['val_loss_mean'] - metrics['val_loss_std'],
                     metrics['val_loss_mean'] + metrics['val_loss_std'],
                     alpha=0.2, color=val_color)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    style_axes(ax1)

    # Plot Accuracy
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Cross-Val Accuracy")
    ax2.plot(epochs, metrics['train_acc_mean'], label='Train Acc', color=train_color, linewidth=line_width)
    ax2.fill_between(epochs,
                     metrics['train_acc_mean'] - metrics['train_acc_std'],
                     metrics['train_acc_mean'] + metrics['train_acc_std'],
                     alpha=0.2, color=train_color)
    ax2.plot(epochs, metrics['val_acc_mean'], label='Val Acc', color=val_color, linewidth=line_width)
    ax2.fill_between(epochs,
                     metrics['val_acc_mean'] - metrics['val_acc_std'],
                     metrics['val_acc_mean'] + metrics['val_acc_std'],
                     alpha=0.2, color=val_color)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    style_axes(ax2)

    plt.tight_layout()
    plt.show()
