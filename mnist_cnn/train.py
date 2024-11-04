import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from .MnistCNN import MnistCNN
from .transforms import get_train_transforms, get_test_transforms


def train_step(model, dataloader, criterion, optimizer, device):
    running_metrics = {"loss": 0.0, "accuracy": 0.0}

    n_batches = len(dataloader)
    progress_bar = tqdm(
        enumerate(dataloader), 
        desc="Training", 
        total=n_batches, 
        unit="batch", 
        colour="green"
    )

    model.train()
    for i, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_metrics["loss"] += loss.item()
        running_metrics["accuracy"] += (outputs.argmax(1) == labels).float().mean().item()

        progress_bar.set_postfix({
            "Loss":     f"{running_metrics['loss'] / (i + 1):.4f}",
            "Accuracy": f"{running_metrics['accuracy'] / (i + 1):.4f}"
        })

    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    return avg_metrics


def test_step(model, dataloader, criterion, device):
    running_metrics = {"loss": 0.0, "accuracy": 0.0}

    n_batches = len(dataloader)
    progress_bar = tqdm(
        enumerate(dataloader), 
        desc="Testing", 
        total=n_batches, 
        unit="batch", 
        colour="blue"
    )

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            running_metrics["loss"] += loss.item()
            running_metrics["accuracy"] += (outputs.argmax(1) == labels).float().mean().item()

            progress_bar.set_postfix({
                "Loss":     f"{running_metrics['loss'] / (i + 1):.4f}",
                "Accuracy": f"{running_metrics['accuracy'] / (i + 1):.4f}"
            })

    avg_metrics = {k: v / n_batches for k, v in running_metrics.items()}
    return avg_metrics


def train_cnn(model):
    epochs = 10
    batch_size = 128
    learning_rate = 3e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Training on {device}")

    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()
    
    train_dataset = MNIST("data", train=True, download=True, transform=train_transforms)
    test_dataset = MNIST("data", train=False, download=True, transform=test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print("-" * 30)

        train_metrics = train_step(model, train_loader, criterion, optimizer, device)
        test_metrics = test_step(model, test_loader, criterion, device)

        print(f"Train Loss: {train_metrics['loss']:.4f} - Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Test Loss:  {test_metrics['loss']:.4f} - Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print()

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn.pth")


if __name__ == "__main__":
    model = MnistCNN()
    train_cnn(model)
