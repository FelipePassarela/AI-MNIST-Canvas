import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fcs = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(128, 10)
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def forward(self, x):
        conv1_featmaps = self.convs[0](x)
        conv2_featmaps = self.convs[1:5](conv1_featmaps)
        conv3_featmaps = self.convs[5:9](conv2_featmaps)
        logits = self.convs[9:](conv3_featmaps)
        logits = self.fcs(logits)
        return logits, (conv1_featmaps, conv2_featmaps, conv3_featmaps)
    
    def train_step(self, dataloader, criterion, optimizer):
        self.train()
        device = next(self.parameters()).device

        total = 0
        correct = 0
        running_loss = 0
        n_batches = len(dataloader)

        for i, (img, label) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)
            pred, _ = self(img)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += label.size(0)
            correct += (pred.argmax(1) == label).sum().item()
            running_loss += loss.item()

            if i % 99 == 0 or i == n_batches - 1:
                train_loss = running_loss / (i + 1)
                print(f"\r\tBatch [{i + 1}/{n_batches}] - Train Loss: {train_loss:.4f}", end='')
        print(f" - Train Accuracy: {100 * correct / total:.2f}%")
        
    def val_step(self, dataloader, criterion):
        self.eval()
        device = next(self.parameters()).device

        correct = 0
        total = 0
        running_loss = 0
        n_batches = len(dataloader)

        with torch.no_grad():
            for i, (img, label) in enumerate(dataloader):
                img, label = img.to(device), label.to(device)
                pred, _ = self(img)
                loss = criterion(pred, label)

                total += label.size(0)
                correct += (pred.argmax(1) == label).sum().item()
                running_loss += loss.item()

                if i % 100 == 0 or i == len(dataloader) - 1:
                    val_loss = running_loss / (i + 1)
                    print(f"\r\tBatch [{i + 1}/{n_batches}] - Val Loss: {val_loss:.4f}", end='')
        print(f" - Val Accuracy: {100 * correct / total:.2f}%")

        accuracy = correct / total
        return loss / len(dataloader.dataset), accuracy

    def load(self, path):
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        self.load_state_dict(torch.load(path))
        self.eval()

    def save(self, path):
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        torch.save(self.state_dict(), path)

    def predict(self, image):
        device = next(self.parameters()).device
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0).to(device)
            output, featmaps = self(image)
            probas = F.softmax(output).squeeze(0).cpu().numpy()
            featmaps = [fm.squeeze(0).cpu().numpy() for fm in featmaps]
        return probas, featmaps


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)

    train_set = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=model.transform)
    test_set = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=model.transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]:")
        model.train_step(train_loader, criterion, optimizer)
        model.val_step(test_loader, criterion)

    model.save("models/cnn.pth")
    