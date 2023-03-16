import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            # input : z (100)
            nn.ConvTranspose2d(100, 64*8, 4, stride=1, padding=0),
            nn.BatchNorm2d(64*8),
            nn.ReLU(),

            # (64*8) * 4 * 4
            nn.ConvTranspose2d(64*8, 64*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),

            # (64*4) * 8 * 8
            nn.ConvTranspose2d(64*4, 64*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),  
            
            # (64*2) * 16 * 16
            nn.ConvTranspose2d(64*2, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64) * 32 * 32
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

        self.initialize_weights()
        
    def forward(self, x):
        out = self.layers(x)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                # torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            # (3) * 64 * 64
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # (64) * 32 * 32
            nn.Conv2d(64, 64*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2),

            # (64*2) * 16 * 16
            nn.Conv2d(64*2, 64*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2),

            # (64*4) * 8 * 8
            nn.Conv2d(64*4, 64*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64*8, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def forward(self, x):
        out = self.layers(x)
        # return out
        return out.view(-1, 1).squeeze(1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                # torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)