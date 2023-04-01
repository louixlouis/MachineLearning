import numpy as np

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

class G_init(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(G_init, self).__init__()
        '''
        Structure : PixelNorm + DenseLayer + LeakyReLU + Conv2d + LeakyReLU + PixelNorm
        Ouput dim : 4*4
        '''
        self.model = nn.Sequential(
            PixelWiseNorm(),
            DeconvBlock(
                in_channels=self.in_channels, 
                out_channels=self.out_channels,
                kernel_size=4,
                stride=1,
                padding=3),
            DeconvBlock(
                in_channels=self.out_channels, 
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
    def forward(self, x):
        return self.model(x)

class G_intermediate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G_intermediate, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            DeconvBlock(
                in_channels=self.in_channels, 
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            DeconvBlock(
                in_channels=self.out_channels, 
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
    def forward(self, x):
        return self.model(x)

class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ToRGB, self).__init__()
        self.model = nn.Sequential(
            EqualizedConv2dLayer(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1, 
                stride=1,
                padding=0),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    '''
    Parameters:
    latent_z_dim : 512 * 1
    c_dim :
    feature_dim :
    activation : relu or leaky relu
    '''
    def __init__(
        self, 
        latent_dim,
        resolution) -> None:
        super(Generator, self).__init__()
        self.num_blocks = 1
        self.alpha = 1
        self.fade_iters = 0

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.model = nn.ModuleList([G_init(in_channels=latent_dim, out_channels=latent_dim)])
        self.to_RGBs = nn.ModuleList([ToRGB(latent_dim, 3)])
        for power in range(2, int(np.log2(resolution))):
            if power < 6:
                in_channels = 512
                out_channels = 512
                self.model.append(G_intermediate(in_channels=in_channels, out_channels=out_channels))
            else:
                in_channels = int(512 / pow(2, power-6))
                out_channels = int(512 / pow(2, power-5))
                self.model.append(G_intermediate(in_channels=in_channels, out_channels=out_channels))
            self.to_RGBs.append(ToRGB(out_channels, 3))

    def grow_model(self):
        print(f'Growing model')
        self.num_blocks += 1

    def forward(self, x):
        for prev_block in self.model[:self.num_blocks-1]:
            x = prev_block(x)
        # Last Generator block
        out = self.model[self.num_blocks-1](x)
        # Convert to RGB
        out = self.to_RGBs[self.num_blocks-1][out]

        # Fade in step.
        if self.alpha < 1:
            old_out = self.up_sample(x)
            old_out = self.to_RGBs[self.num_blocks-2][old_out]
            out = (1 - self.alpha)*old_out + self.alpha*out

            # 여기 계산 방식 이해 안감
            self.alpha += self.fade_iters
        return out

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

