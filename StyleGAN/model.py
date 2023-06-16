import torch
import torch.nn as nn
import torch.nn.functional as F

from customLayers import *

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim) -> None:
        super(MappingNetwork, self).__init__()
        self.layers = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )
    
    def forward(self, x):
        return self.layers(x)
    
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim) -> None:
        super(GenBlock, self).__init__()
        self.conv_1 = WSConv2d(in_channels=in_channels, out_channels=out_channels)
        self.l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.noise_1 = AddNoise(out_channels)
        self.adaIN_1 = AdaIN(out_channels=out_channels, w_dim=w_dim)
        
        self.conv_2 = WSConv2d(in_channels=out_channels, out_channels=out_channels)
        self.noise_2 = AddNoise(out_channels)
        self.adaIN_2 = AdaIN(out_channels=out_channels, w_dim=w_dim)
    
    def forward(self, x, w):
        out = self.adaIN_1(self.l_relu(self.noise_1(self.conv_1(x))), w)
        out = self.adaIN_2(self.l_relu(self.noise_2(self.conv_2(out))), w)
        return out
    
class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, in_channels, factors, img_channels=3) -> None:
        '''
        z_dim
        w_dim
        in_channels
        factors = [1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
        '''
        super(Generator, self).__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        # Why ones?
        self.constant = nn.Parameter(torch.ones(1, in_channels, 4, 4))

        # Initial block.
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.init_l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.init_noise_1 = AddNoise(in_channels)
        self.init_adaIN_1 = AdaIN(in_channels, w_dim)

        self.init_noise_2 = AddNoise(in_channels)
        self.init_adaIN_2 = AdaIN(in_channels, w_dim)

        self.to_rgb = WSConv2d(in_channels=in_channels, out_channels=img_channels, kernel_size=1, stride=1, padding=0)
        
        self.gen_block_list = nn.ModuleList([])
        self.to_rgb_list = nn.ModuleList([self.to_rgb])

        for i in range(len(factors)-1):
            in_c = int(in_channels * factors[i])
            out_c = int(in_channels * factors[i+1])
            self.gen_block_list.append(GenBlock(in_channels=in_c, out_channels=out_c, w_dim=w_dim))
            self.to_rgb_list.append(WSConv2d(in_channels=in_c, out_channels=img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, up_sampled, generated):
        return torch.tanh(alpha*generated + (1-alpha)*up_sampled)

    def forward(self, x, alpha, steps):
        w = self.mapping(x)

        # Initial block
        out = self.init_adaIN_1(self.init_noise_1(self.constant), w)
        out = self.init_adaIN_2(self.init_l_relu(self.init_noise_2(self.init_conv(out))), w)

        if steps == 0:
            # x (output of init_adaIN_1) ? 
            # out (output of init_adaIN_2) ?
            return self.to_rgb_list[steps](x)

        for step in range(steps):
            up_sampled = F.interpolate(out, scale_factor=2, mode='bilinear')
            out = self.gen_block_list[step](up_sampled, w)
        
        final_up_sampled = self.to_rgb_list[steps-1](up_sampled)
        final_out = self.to_rgb_list[steps](out)

        return self.fade_in(alpha=alpha, up_sampled=up_sampled, generated=final_out)