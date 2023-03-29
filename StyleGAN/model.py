import torch
import torch.nn as nn

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
    Mapping Layer
    '''
    def __init__(self, z_dim, activation, num_layers) -> None:
        super(MappingLayer, self).__init__()
        self.z_dim = z_dim
        self.num_layers = num_layers

        for num in range(self.num_layers):
            layer = FullyConnectedLayer(512, 512, activation=activation)
    def forward(self, x):
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