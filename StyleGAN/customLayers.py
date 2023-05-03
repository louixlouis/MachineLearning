import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelWiseNorm(nn.Module):
    def __init__(self) -> None:
        super(PixelWiseNorm, self).__init__()
        self.eps = 1e-8
    def forward(self, x):
        # What is the difference bewteen sqrt and rsqrt.
        return x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.eps)

class MinibatchSTD(nn.Module):
    def __init__(self):
        super(MinibatchSTD, self).__init__()
    def forward(self, x):
        # 이게 왜 minbatch std?
        size = list(x.size())
        size[1] = 1
        std = torch.std(x, dim=0)
        std_mean = torch.mean(std)
        return torch.cat((x, std_mean.repeat(size)), dim=1)

class UpScale2d(nn.module):
    '''
    Upsampling 쓰면 안되나?
    '''
    def upscale2d():
        pass

class DownScale2d(nn.module):
    '''
    Downsampling 쓰면 안되나?
    '''
    def downscale2d():
        pass


class EqualizedConv2d(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, paddding=0):
        super(EqualizedConv2d, self).__init__()
    def forward(self, x):
        return x

class NoiseLayer(nn.module):
    '''
    '''
    def __init__(self, out_channels):
        super(NoiseLayer, self).__init__()
    
    def forward(self, x):
        return x
    
class DeconvBlock(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DeconvBlock, self).__init__()
    def forward(self, x):
        return x

class ConvBlock(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
    def forwrad(self, x):
        return x

        