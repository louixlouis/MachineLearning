import torch
import torch.nn as nn

class Swish(nn.Module):
    '''
    Swish activation function
        x * sigmoid(x)
    
    nn.Sigmoid      -> class
    torch.sigmoid   -> function
    '''
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self):
        super(TimeEmbedding, self).__init__()
    
    def forward(self, x):
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

    def forwrad(self, x):
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self) -> None:
        super(AttentionBlock, self).__init__()

    def forward(self, x):
        return x
