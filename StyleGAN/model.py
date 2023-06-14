import torch
import torch.nn as nn

# AffineTransform X, 이름변경 필요.
class AffineTransform(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(AffineTransform, self).__init__()
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.scale = (2 / in_channels) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        # Initialization
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(self.scale * x) + self.bias

class PixelNorm(nn.Moduel):
    def __init__(self) -> None:
        super(PixelNorm, self).__init__()
        self. eps = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)

class AdaIN(nn.Module):
    def __init__(self, w_dim, out_channels) -> None:
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.style_scale = AffineTransform(in_channels=w_dim, out_channels=out_channels)
        self.style_bias = AffineTransform(in_channels=w_dim, out_channels=out_channels)
    
    def forward(self, x, w):
        out = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale*out + style_bias
    
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim) -> None:
        super(MappingNetwork, self).__init__()
        self.layers = nn.Sequential(
            PixelNorm(),
            AffineTransform(z_dim, w_dim),
            nn.ReLU(),
            AffineTransform(w_dim, w_dim),
            nn.ReLU(),
            AffineTransform(w_dim, w_dim),
            nn.ReLU(),
            AffineTransform(w_dim, w_dim),
            nn.ReLU(),
            AffineTransform(w_dim, w_dim),
            nn.ReLU(),
            AffineTransform(w_dim, w_dim),
            nn.ReLU(),
            AffineTransform(w_dim, w_dim),
            nn.ReLU(),
            AffineTransform(w_dim, w_dim),
        )
    
    def forward(self, x):
        return self.layers(x)