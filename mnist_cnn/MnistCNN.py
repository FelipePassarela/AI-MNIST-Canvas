import numpy as np
import torch
import torch.nn as nn

from .transforms import get_inference_transforms


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.GELU(),

            nn.Linear(256, 10)
        )

    def forward(self, input):
        featmaps = []
        for layer in self.main:
            input = layer(input)
            if isinstance(layer, nn.Conv2d):
                featmaps.append(input.detach().clone())
        return input, featmaps

    def predict(self, image: np.ndarray):
        """
        Predicts the class probabilities and feature maps for a given input image.

        Args:
            image (np.ndarray): Input image array of shape [H, W].

        Returns:
            tuple:
                - probas (np.ndarray): Array of class probabilities for each class.
                - featmaps (list): List of feature maps extracted from the network layers.
        """

        transforms = get_inference_transforms()
        device = next(self.parameters()).device

        with torch.no_grad():
            image = transforms(image).to(device).unsqueeze(0)
            output, featmaps = self(image)
            probas = torch.softmax(output, dim=-1).squeeze(0).cpu().numpy()
            featmaps = [fm.squeeze(0).cpu().numpy() for fm in featmaps]
        return probas, featmaps
