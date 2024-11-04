import torch
from torchvision.transforms import v2 as T


def get_train_transforms():
    return T.Compose([
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        T.RandomPerspective(distortion_scale=0.2, p=0.2),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize((0.1307,), (0.3081,)),
        T.RandomErasing(p=0.1)
    ])


def get_test_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
        T.Lambda(lambda x: x + torch.randn_like(x) * 0.1)  # Gaussian noise
    ])


def get_inference_transforms():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize((0.1307,), (0.3081,))
    ])
