import torch
import torch.nn as nn

class PixelWiseNorm(nn.Module):
    def __init__(self) -> None:
        super(PixelWiseNorm, self).__init__()
        self.eps = 1e-8
    def forward(self, x):
        out = x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps)**0.5
        return out
    
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
    
class EqualizedConvolutionLayer(nn.Module):
    '''
    Equalized Convolutional Layer
    For equalized learning rate
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, initializer='kaiming') -> None:
        super(EqualizedConvolutionLayer, self).__init__()
        self.layer = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias)
        
        # Initialize.

        weights = self.layer.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))
        self.scale = (torch.mean(self.layer.weight.data**2))**0.5
        self.layer.weight.data.copy_(self.layer.weight.data/self.scale)

    def forward(self, x):
        out = self.layer(x.mul(self.scale))
        return out + self.bias.view(1, -1, 1, 1).expand_as(out)
    
class DenseLayer(nn.Module):
    '''
    Dense Layer
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, ) -> None:
        super(DenseLayer, self).__init__()
        layers = []
        if True:
            pass
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.layers = nn.Sequential(layers)

    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, latent_z_dim, c_dim, feature_dim, batch_norm, ) -> None:
        '''
        Generator

        '''
        super(Generator, self).__init__()
        self.latent_z_dim = latent_z_dim
        self.c_dim = c_dim
        self.feature_dim = feature_dim
        self.batch_norm = batch_norm

        layers = []
        # First Block
        layers.append(PixelWiseNorm())
    def forward(self, x):
        return x

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

