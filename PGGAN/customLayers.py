import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class PixelWiseNorm(nn.Module):
    def __init__(self) -> None:
        super(PixelWiseNorm, self).__init__()
        self.eps = 1e-8
    def forward(self, x):
        return x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.eps)
        
class MinibatchSTD(nn.Module):
    def __init__(self):
        super(MinibatchSTD, self).__init__()
    def forward(self, x):
        size = list(x.size())
        size[1] = 1
        std = torch.std(x, dim=0)
        std_mean = torch.mean(std)
        return torch.cat((x, std_mean.repeat(size)), dim=1)

class EqualizedConv2dLayer(nn.Module):
    '''
    Equalized Convolutional Layer
    For equalized learning rate
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(EqualizedConv2dLayer, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.scale = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        
        # Initialization
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)
    def forward(self, x):
        return F.conv2d(
            input=x, 
            weight=self.weight*self.scale, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding)
    
class DeconvBlock(nn.Module):
    '''
    Structure
    Init Block  : PixelNorm + Conv2d + LeakyReLU + PixelNorm
    Other Block : Conv2d + LeakyReLU + PixelNorm

    Question :
    Why all guys use EqualizedConv2dLayer instead of Dense layer?
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DeconvBlock, self).__init__()
        self.model = nn.Sequential(
            EqualizedConv2dLayer(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride,
                padding=padding),
            nn.LeakyReLU(0.2),
            PixelWiseNorm())

    def forward(self, x):
        return self.model(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            EqualizedConv2dLayer(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride,
                padding=padding),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.model(x)