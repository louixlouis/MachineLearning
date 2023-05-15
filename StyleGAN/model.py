import numpy as np

import torch
import torch.nn as nn

from customLayers import PixelWiseNorm, EqualizedLinear, EqualizedConv2d

class InputBlock(nn.Module):
    '''
    The first block (4x4 pixels) doesn't hava an input.
    The result of it is just replaced by a trained constant.
    '''
    def __init__(
            self,
            in_channels,
            latent_w_dim,
            gain,
            ) -> None:
        super(InputBlock, self).__init__()

        self.in_channels = in_channels
        self.constant = nn.Parameter(torch.ones(1, self.in_channels, 4, 4))
        self.bias = nn.Parameter(torch.ones(in_channels))

        self.epi_1 = None
        self.conv2d = EqualizedConv2d()
        self.epi_2 = None
class MappingLayer(nn.Module):
    '''
    Mapping Layer.
    '''
    def __init__(
            self, 
            latent_z_dim, 
            latent_w_dim, 
            latent_w_broadcast, 
            num_layers=8, 
            lr_multiplier=0.01, 
            use_relu='lrelu', 
            normalize_z=True, 
            w_scale=True
            ) -> None:
        '''
        Parameters :
        latent_z_dim       : dimension of latent vector z.
        latent_w_dim       : dimension of disentangled latent w.
        latent_w_broadcast : broadcast latent vector w as [minibatch, latent_w_broadcast] or [minibatch, latent_w_broadcast, latent_w_dim]
        num_layers         : number of mapping layers.
        lr_multiplier      : learning rate multiplier.
        use_relu           : activation functions, 'relu' or 'lrelu'
        normalize_z        : normalize letent vector z before feeding to layers.
        w_scale            : enable equalized learning rate
        '''
        super(MappingLayer, self).__init__()
        self.latent_z_dim = latent_z_dim
        self.latent_w_dim = latent_w_dim
        self.latent_w_broadcast = latent_w_broadcast
        self.num_layers = num_layers
         
        # Activation functions.
        if use_relu:
            activation_layer = nn.ReLU()
        else:
            activation_layer = nn.LeakyReLU(0.2)

        # Mapping layers.
        layers = []
        if normalize_z:
            layers.append(PixelWiseNorm())
        layers.append(
            EqualizedLinear(
                in_channels=self.latent_z_dim, 
                out_channels=self.latent_w_dim, 
                gain=np.sqrt(2), 
                w_scale=w_scale, 
                lr_multiplier=lr_multiplier))
        layers.append(activation_layer)

        for num in range(1, self.num_layers):
            layers.append(
                EqualizedLinear(
                    in_channels=self.latent_w_dim,
                    out_channels=self.latent_w_dim,
                    gain=np.sqrt(2),
                    w_scale=w_scale,
                    lr_multiplier=lr_multiplier
                ))
            layers.append(activation_layer)
        self.model = nn.Sequential(*layers)
                      
    def forward(self, x):
        w = self.model(x)
        if self.latent_w_broadcast is not None:
            w = w.unsqueeze(1).expand(-1, self.latent_w_broadcast, -1)
        return w

class SynthesizeLayer(nn.Module):
    def __init__(
            self,
            latent_w_dim=512,
            num_channels=3,
            resolution=1024,
            use_styles=True,
            use_relu=False,
            structure='linear'
            ) -> None:
        '''
        resolution : Output resolution
        use_styles : Enable style inputs
        use_relu : True -> relu, False -> leaky relu
        structure : 'fixed' -> no progressive, 'linear' -> human readable?
        '''
        super(SynthesizeLayer, self).__init__()
        self.structure = structure
        
        log2_resolution = int(np.lgo2(resolution))
        assert resolution == 2**log2_resolution and resolution >= 4
        self.depth = log2_resolution - 1

        # What different between self.depth and self.num_layers?
        self.num_layers = log2_resolution*2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        gain = np.sqrt(2)
        if use_relu:
            activation_layer = nn.ReLU()
        else:
            activation_layer = nn.LeakyReLU(0.2)
        
        # Early layer
        self.input_block = InputBlock()

        # to_RGB layers for various outputs.

    def forward(self, x):
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Mapping Network.
        # Consist of 8 fc layers
        self.latent_z_dim = 512
        self.latent_w_dim = 512

        self.mapping_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512)
        )

        self.synthesize_layer = nn.Sequential(

        )

    def mapping(self, x):
        out = self.mapping_layers(x)
        return out

    def synthesize(self, x):
        out = self.synthesize_layers(x)
        return out
    
    def forward(self, x):
        pass

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):
        out = x
        return out
