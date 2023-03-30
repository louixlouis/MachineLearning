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
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 bias=False, 
                 initializer='kaiming') -> None:
        super(EqualizedConvolutionLayer, self).__init__()
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
    
class ConvBlock(nn.Module):
    '''
    Conv Block
    '''
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0,
                 activation='leaky_relu', 
                 weight_norm=False,
                 batch_norm=False,
                 pixel_norm=False,
                 block=True) -> None:
        super(ConvBlock, self).__init__()
        layers = []
        if weight_norm:
            layers.append(EqualizedConvolutionLayer(in_channels=in_channels, 
                                                    out_channels=out_channels, 
                                                    kernel_size=kernel_size, 
                                                    stride=stride,
                                                    padding=padding))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        
        if block:
            if activation=='leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BarchNorm2d(out_channels))
            if pixel_norm:
                layers.append(PixelWiseNorm())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, 
                 latent_z_dim, 
                 c_dim, 
                 feature_dim,
                 activation,
                 weight_norm, 
                 batch_norm,
                 pixel_norm) -> None:
        '''
        Generator

        '''
        super(Generator, self).__init__()
        self.in_channels = latent_z_dim
        self.c_dim = c_dim
        self.feature_dim = feature_dim

        self.activation = activation
        self.weight_norm = weight_norm
        self.batch_norm = batch_norm
        self.pixel_norm= pixel_norm

        # self.model = nn.Sequential(self.first_block() + self.intermediate_block() +self.to_rgb_block())
        self.model = nn.Sequential()
        self.model.add_module('first_block', self.first_block())
        self.model.add_module('to_rgb_block', self.to_rgb_block())

    def first_block(self):
        layers = []
        # First Block
        layers.append(PixelWiseNorm())
        layers.append(ConvBlock(in_channels=self.in_channels, 
                                out_channels=self.feature_dim,
                                kernel_size=4,
                                stride=1,
                                padding=3,
                                activation=self.activation,
                                weight_norm=self.weight_norm,
                                batch_norm=self.batch_norm,
                                pixel_norm=self.pixel_norm))
        layers.append(ConvBlock(in_channels=self.feature_dim, 
                                out_channels=self.feature_dim,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                activation=self.activation,
                                weight_norm=self.weight_norm,
                                batch_norm=self.batch_norm,
                                pixel_norm=self.pixel_norm))
        return nn.Sequential(*layers)
        # return layers
    
    def intermediate_block(self):
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(ConvBlock(in_channels=self.feature_dim, 
                                out_channels=self.feature_dim,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                activation=self.activation,
                                weight_norm=self.weight_norm,
                                batch_norm=self.batch_norm,
                                pixel_norm=self.pixel_norm))
        layers.append(ConvBlock(in_channels=self.feature_dim, 
                                out_channels=self.feature_dim,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                activation=self.activation,
                                weight_norm=self.weight_norm,
                                batch_norm=self.batch_norm,
                                pixel_norm=self.pixel_norm))
        return nn.Sequential(*layers)
        # return layers
    
    def to_rgb_block(self):
        layers = []
        layers.append(ConvBlock(in_channels=self.feature_dim, 
                                out_channels=self.feature_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                weight_norm=self.weight_norm,
                                block=False))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)
        # return layers
    
    def grow_model(self, resl):
        '''
        Initial value of resl is 2 (2*2 starts)

        '''
        new_model = nn.Sequential()
        for name, module in self.model.named_children():
            if not name=='to_rgb_block':
                new_model.add_module(name, module)                  # Add module
                new_model[-1].load_state_dict(module.state_dict())  # Copy trained weights

        if resl >= 3 and resl <=9:
            print(f'Growing network[{int(pow(2, resl-1))}X{int(pow(2, resl-1))} to {int(pow(2, resl))}X{int(pow(2, resl))}]')

    def forward(self, x):
        out = self.model(x.view(x.shape[0], -1, 1, 1))
        return out

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

