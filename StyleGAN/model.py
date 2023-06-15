import torch
import torch.nn as nn

# AffineTransform X, 이름변경 필요.
class WSLinear(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.scale = (2 / in_channels) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # Initialization
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(self.scale * x) + self.bias

class PixelNorm(nn.Module):
    def __init__(self) -> None:
        super(PixelNorm, self).__init__()
        self. eps = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)

class AdaIN(nn.Module):
    def __init__(self, w_dim, out_channels) -> None:
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.style_scale = WSLinear(in_channels=w_dim, out_channels=out_channels)
        self.style_bias = WSLinear(in_channels=w_dim, out_channels=out_channels)
    
    def forward(self, x, w):
        out = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale*out + style_bias

class AddNoise(nn.Module):
    def __init__(self, channels) -> None:
        super(AddNoise, self).__init__()   
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        noise = torch.randn((x.shape[0], x.shape[1], x.shape[2]), device=x.device)
        # 계산식 맞음?
        return x + self.weight + noise
    
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1) -> None:
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialization
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(self.scale * x) + self.bias.view(1, self.bias.shape[0], 1, 1)
    
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
    def __init__(self, z_dim, w_dim, in_channels, img_channels=3) -> None:
        super(Generator, self).__init__()
        # Why ones?
        self.constant = nn.Parameter(torch.ones(1, in_channels, 4, 4))
        self.mapping = MappingNetwork(z_dim, w_dim)

        # Initial block.
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.init_l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.init_noise_1 = AddNoise(in_channels)
        self.init_adaIN_1 = AdaIN(in_channels, w_dim)

        self.init_noise_2 = AddNoise(in_channels)
        self.init_adaIN_2 = AdaIN(in_channels, w_dim)

