import torch
import torch.nn as nn

class UpScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x

class DownScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class ConvolutionalLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class FullyConnectedLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

