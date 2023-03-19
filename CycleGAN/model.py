import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, feature_dim):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3),
            nn.InstanceNorm2d(feature_dim),
            nn.ReflectionPad2d(1),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3),
            nn.InstanceNorm2d(feature_dim),
        )
    
    def forward(self, x):
        out = x + self.layers(x)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Input layer

        # Encoding
        # Convolution layers
        feature_dim = 64
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(feature_dim*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_dim*2, feature_dim*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(feature_dim*4),
            nn.ReLU(inplace=True),
        )

        # Intermediate
        # Residual layers
        self.inter_layers = nn.Sequential(
            ResidualBlock(feature_dim*4),
            ResidualBlock(feature_dim*4),
            ResidualBlock(feature_dim*4),
            ResidualBlock(feature_dim*4),
            ResidualBlock(feature_dim*4),
            ResidualBlock(feature_dim*4),
            ResidualBlock(feature_dim*4),
            ResidualBlock(feature_dim*4),
            ResidualBlock(feature_dim*4),
        )
        # self.residual_layer_1 = ResidualBlock(feature_dim*4)
        # self.residual_layer_2 = ResidualBlock(feature_dim*4)
        # self.residual_layer_3 = ResidualBlock(feature_dim*4)
        # self.residual_layer_4 = ResidualBlock(feature_dim*4)
        # self.residual_layer_5 = ResidualBlock(feature_dim*4)
        # self.residual_layer_6 = ResidualBlock(feature_dim*4)
        # self.residual_layer_7 = ResidualBlock(feature_dim*4)
        # self.residual_layer_8 = ResidualBlock(feature_dim*4)
        # self.residual_layer_9 = ResidualBlock(feature_dim*4)

        # Decoding
        # Deconvolution layers
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*4, feature_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(feature_dim*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_dim*2, feature_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Output layer

    def forward(self, x):
        out = self.encoder_layers(x)
        out = self.inter_layers(out)
        out = self.decoder_layers(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        feature_dim = 64
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim, feature_dim*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(feature_dim*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim*2, feature_dim*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(feature_dim*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim*4, feature_dim*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(feature_dim*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_dim*8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = F.avg_pool2d(out, out.size()[2:]).view(out.shape[0], -1)
        return out