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
        self.conv1 = nn.Conv2d(1, 32, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def forward(self, x):
        conv1_featmaps = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(conv1_featmaps, 2)
        conv2_featmaps = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(conv2_featmaps, 2)
        
        x = torch.flatten(x, 1)
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x, conv1_featmaps, conv2_featmaps
    
    def train_model(self, train_loader, test_loader, epochs=10, learning_rate=0.001):
        self.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for image, label in train_loader:
                image, label = image.to(self.device), label.to(self.device)
                output, _, _ = self(image)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, ", end="")
            self.evaluate(test_loader)

        os.makedirs("models", exist_ok=True)
        torch.save(self.state_dict(), "models/cnn.pth")

    def evaluate(self, test_loader):
        self.to(self.device)
        self.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for image, label in test_loader:
                image, label = image.to(self.device), label.to(self.device)
                output, _, _ = self(image)
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        print(f"Accuracy: {correct / total:.4f}")

    def load(self, path):
        dir = os.path.dirname(path)
        os.makedirs(dir, exist_ok=True)
        self.load_state_dict(torch.load(path))
        self.to(self.device)
        self.eval()
        print(f"Model loaded with device: {self.device}")

    def predict(self, image):
        self.to(self.device)
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0).to(self.device)
            output, conv1_featmaps, conv2_featmaps = self(image)
            probas = torch.exp(output)
            return (
                probas.squeeze().tolist(), 
                conv1_featmaps.squeeze().tolist(), 
                conv2_featmaps.squeeze().tolist()
            )


if __name__ == '__main__':
    cnn = CNN()

    train_set = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=cnn.transform)
    test_set = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=cnn.transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    cnn.train_model(train_loader, test_loader)
    