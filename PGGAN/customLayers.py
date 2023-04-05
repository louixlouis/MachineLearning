import torch
import torch.nn as nn

class PixelWiseNorm(nn.Module):
    def __init__(self) -> None:
        super(PixelWiseNorm, self).__init__()
        self.eps = 1e-8
    def forward(self, x):
        out = x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps)**0.5
        return out
    
class EqualizedConv2dLayer(nn.Module):
    '''
    Equalized Convolutional Layer
    For equalized learning rate
    '''
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 bias=False, 
                 initializer='kaiming') -> None:
        super(EqualizedConv2dLayer, self).__init__()
        self.layer = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias)
        
        # Initialize.
        if initializer == 'kaiming':
            nn.init.kaiming_uniform_(self.layer.weight, a=nn.init.calculate_gain('conv2d'))
        elif initializer == 'xavier':
            nn.init.xavier_uniform_(self.layer.weight)

        # weights = self.layer.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))
        self.scale = (torch.mean(self.layer.weight.data**2))**0.5
        self.layer.weight.data.copy_(self.layer.weight.data/self.scale)

    def forward(self, x):
        out = self.layer(x.mul(self.scale))
        return out + self.bias.view(1, -1, 1, 1).expand_as(out)
    
class DeconvBlock(nn.Module):
    '''
    Structure
    Init Block  : PixelNorm + Conv2d + LeakyReLU + PixelNorm
    Other Block : Conv2d + LeakyReLU + PixelNorm

    Question :
    Why all guys use EqualizedConv2dLayer instead of Dense layer?
    '''
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0) -> None:
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

        # Need to add weight initialization.
    def forward(self, x):
        return self.model(x)

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

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0) -> None:
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