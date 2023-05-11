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
    def __init__(self) -> None:
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


class EqualizedLinear(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels, bias=True, gain=2**0.5, w_scale=False, lr_multiplier=1) -> None:
        super(EqualizedLinear, self).__init__()
        # He initialization.
        He_std = gain * in_channels**(-0.5)
        if w_scale:
            init_std = 1.0 / lr_multiplier
            self.w_mul = He_std * lr_multiplier
        else :
            init_std = He_std / lr_multiplier
            self.w_mul = lr_multiplier
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_mul = lr_multiplier
        else :
            self.bias = None
            
    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        out = F.linear(x, self.weight * self.w_mul, bias)
        return out

class EqualizedConv2d(nn.Module):
    def __init__(self) -> None :
        pass
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

        
