import numpy as np

import torch
import torch.nn as nn

from customLayers import *

####
# Generator
####
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
                in_channels=in_channels, 
                out_channels=in_channels,
                kernel_size=4,
                stride=1,
                padding=3),
            DeconvBlock(
                in_channels=in_channels, 
                out_channels=out_channels,
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
                in_channels=in_channels, 
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            DeconvBlock(
                in_channels=in_channels, 
                out_channels=out_channels,
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
    latent_z_dim : 
    '''
    def __init__(self, latent_dim, resolution) -> None:
        super(Generator, self).__init__()
        self.num_blocks = 1     # Number of generator blocks (G_init, G_int, G_int, ...)
        self.alpha = 1
        self.alpha_step = 0

        # Initial Structure
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.model = nn.ModuleList([G_init(in_channels=latent_dim, out_channels=latent_dim)])
        self.to_RGBs = nn.ModuleList([ToRGB(latent_dim, 3)])

        # Add intermediate blocks.
        for power in range(2, int(np.log2(resolution))):
            if power < 6:
                '''
                512x8x8 -> 512x16x16 -> 512x32x32
                '''
                in_channels = 512
                out_channels = 512
                self.model.append(G_intermediate(in_channels=in_channels, out_channels=out_channels))
            else:
                '''
                channels?
                64x64 -> 128x128 -> 256x256 -> 512x512 -> 1024x1024
                '''
                in_channels = int(512 / pow(2, power-6))
                out_channels = int(512 / pow(2, power-5))
                self.model.append(G_intermediate(in_channels=in_channels, out_channels=out_channels))
            self.to_RGBs.append(ToRGB(out_channels, 3))

    def grow_model(self, inverse_step):
        self.alpha_step = 1 / inverse_step
        self.alpha = self.alpha_step
        self.num_blocks += 1

        print(f'Growing generator : {pow(2, self.num_blocks+1)}x{pow(2, self.num_blocks+1)} to {pow(2, self.num_blocks+2)}x{pow(2, self.num_blocks+2)}')

    def forward(self, x):
        for prev_block in self.model[:self.num_blocks-1]:
            x = prev_block(x)
        # Last Generator block
        out = self.model[self.num_blocks-1](x)
        # Convert to RGB
        out = self.to_RGBs[self.num_blocks-1](out)

        # Fade in step.
        if self.alpha < 1:
            old_out = self.up_sample(x)
            old_out = self.to_RGBs[self.num_blocks-2](old_out)
            out = (1 - self.alpha)*old_out + self.alpha*out

            self.alpha += self.alpha_step
        return out

####
# Discriminator
####
class D_init(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(D_init, self).__init__()
        '''
        Structure : PixelNorm + DenseLayer + LeakyReLU + Conv2d + LeakyReLU + PixelNorm
        Ouput dim : 4*4
        '''
        self.model = nn.Sequential(
            MinibatchSTD(),
            ConvBlock(
                in_channels=in_channels + 1, # why? 
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            ConvBlock(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=4,
                stride=1,
                padding=0),
            # It need to replace Linear to Dense Conv2d
            nn.Flatten(),
            nn.Linear(out_channels, 1)
            )
    def forward(self, x):
        return self.model(x)

class D_intermediate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(D_intermediate, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(
                in_channels=in_channels, 
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(0.2),
            ConvBlock(
                in_channels=out_channels, 
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.model(x)

class FromRGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FromRGB, self).__init__()
        self.model = nn.Sequential(
            EqualizedConv2dLayer(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, latent_dim, resolution) -> None:
        super(Discriminator, self).__init__()
        self.num_blocks = 1
        self.alpha = 1
        self.alpha_step = 0

        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.model = nn.ModuleList([D_init(in_channels=latent_dim, out_channels=latent_dim)])
        self.from_RGBs = nn.ModuleList([FromRGB(3, latent_dim)])
        for power in range(2, int(np.log2(resolution))):
            if power < 6:
                in_channels = 512
                out_channels = 512
                self.model.append(D_intermediate(in_channels=in_channels, out_channels=out_channels))
            else:
                in_channels = int(512 / pow(2, power-5))
                out_channels = int(512 / pow(2, power-6))
                self.model.append(D_intermediate(in_channels=in_channels, out_channels=out_channels))
            self.from_RGBs.append(FromRGB(3, in_channels))

    def grow_model(self, inverse_step):
        self.alpha_step = 1 / inverse_step
        self.alpha = self.alpha_step
        self.num_blocks += 1

        print(f'Growing discriminator : {pow(2, self.num_blocks+1)}x{pow(2, self.num_blocks+1)} to {pow(2, self.num_blocks+2)}x{pow(2, self.num_blocks+2)}')

    def forward(self, x):
        out = self.from_RGBs[self.num_blocks-1](x)
        out = self.model[self.num_blocks-1](out)
        if self.alpha < 1:
            old_out = self.down_sample(x)
            old_out = self.from_RGBs[self.num_blocks-2](old_out)
            out = (1 - self.alpha)*old_out + self.alpha*out
            self.alpha += self.alpha_step

        for prev_block in reversed(self.model[:self.num_blocks-1]):
            out = prev_block(out)
        return out

