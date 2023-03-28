import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.x_input_layer = nn.Sequential(
            # input : z (100)
            nn.ConvTranspose2d(100, 64*4, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU()
        )

        self.y_input_layer = nn.Sequential(
            # input : z (100)
            nn.ConvTranspose2d(2, 64*4, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            # (64*16) * 4 * 4
            nn.ConvTranspose2d(64*8, 64*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),

            # (64*8) * 8 * 8
            nn.ConvTranspose2d(64*4, 64*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),  
            
            # (64*2) * 16 * 16
            nn.ConvTranspose2d(64*2, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64) * 32 * 32
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.initialize_weights()

    def forward(self, x, y):
        x = self.x_input_layer(x)
        y = y.reshape([y.shape[0], -1, 1, 1])
        y = self.y_input_layer(y)
        out = torch.cat([x, y], dim=1)
        out = self.layers(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.x_input_layer = nn.Sequential(
            # 64 * 3 * 64 * 64
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.y_input_layer = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.layers = nn.Sequential(
            # Concat : 64 * 64 * 32 * 32 and 64 *  
            nn.Conv2d(64, 64*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2),

            # (64*2) * 16 * 16
            nn.Conv2d(64*2, 64*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2),

            # (64*4) * 8 * 8
            nn.Conv2d(64*4, 64*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64*8, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def forward(self, x, y):
        x = self.x_input_layer(x)
        y = self.y_input_layer(y)
        out = torch.cat([x, y], dim=1)
        out = self.layers(out)
        return out.view(-1, 1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)