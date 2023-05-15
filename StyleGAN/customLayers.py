import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurLayer(nn.Module):
    '''
    what does do this Layer?
    '''
    def __init__(self, kernel, stride=1, normalize=True, flip=False) -> None:
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]

        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
            self.register_buffer('kernel', kernel)
            self.stride = stride

    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        out = F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2), groups=x.size(1))
        return out
    
class PixelWiseNorm(nn.Module):
    def __init__(self) -> None:
        super(PixelWiseNorm, self).__init__()
        self.eps = 1e-8
    def forward(self, x):
        # What is the difference bewteen sqrt and rsqrt.
        # return x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.eps)
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)

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
    def __init__(self, factor=2, gain=1) -> None:
        super(UpScale2d, self).__init__()
        self.factor = factor
        self.gain = gain
    
    def forward(self, x):
        out = x * self.gain
        shape = out.shape
        out = out.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
        out = out.contiguous().view(shape[0], shape[1], self.factor*shape[2], self.factor*shape[3])
        return out
    
    @staticmethod
    def upscale2d():
        pass

class DownScale2d(nn.module):
    '''
    Downsampling 쓰면 안되나?
    '''
    def __init__(self, factor=2, gain=1) -> None:
        super(DownScale2d, self).__init__()
        self.factor = factor
        self.gain = gain
        if self.factor == 2:
            kernel = [np.sqrt(self.gain) / self.factor] * self.factor
            self.blur_layer = BlurLayer(kernel=kernel, normalize=False, stride=self.factor)
        else:
            self.blur_layer = None
    def forward(self, x):
        # downscale by using blur_layer
        if self.blur_layer is not None:
            return self.blur_layer(x)
        
        out = x * self.gain
        if self.factor == 1:
            return out
        return F.avg_pool2d(x, self.factor)

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

        
