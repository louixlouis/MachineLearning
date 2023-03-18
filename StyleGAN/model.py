import torch
import torch.nn as nn

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