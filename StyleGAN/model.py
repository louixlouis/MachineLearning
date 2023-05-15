import numpy as np

import torch
import torch.nn as nn

from customLayers import PixelWiseNorm, EqualizedLinear

class FullyConnectedLayer(nn.Module):
    '''
    Customized Fully connected layer
    '''
    def __init__(self, in_feature, out_feature, activation) -> None:
        super(FullyConnectedLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn([out_feature, in_feature]))
        self.bias = True,
        self.activation = activation

    def forward(self, x):
        weight = self.weight.to(x.dtype)

        out = x.matmul(weight.t())
        return out

class ConvolutionalLayer(nn.Module):
    '''
    Cutomized Convolutional Layer
    '''
    def __init__(self, in_channels, out_channels, kernel_size, bias, activation) -> None:
        super(ConvolutionalLayer, self).__init__()
        self.activation = activation
    
    def forward(self, x):
        return x
    
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
            activation='lrelu', 
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
        activation         : activation functions, 'relu' or 'lrelu'
        normalize_z        : normalize letent vector z before feeding to layers.
        w_scale            : enable equalized learning rate
        '''
        super(MappingLayer, self).__init__()
        self.latent_z_dim = latent_z_dim
        self.latent_w_dim = latent_w_dim
        self.latent_w_broadcast = latent_w_broadcast
        self.num_layers = num_layers
         
        # Activation functions.
        if activation == 'lrelu':
            activation_layer = nn.LeakyReLU(0.2)
        else:
            activation_layer = nn.ReLU()

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
        out = self.model(x)
        if self.latent_w_broadcast is not None:
            out = out.unsqueeze(1).expand(-1, self.latent_w_broadcast, -1)
        return x

class SynthesizeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

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
