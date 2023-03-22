import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.x_input_layer = nn.Sequential(
            nn.ConvTranspose2d(100, 64*2, 3, stride=1, padding=0),
            nn.BatchNorm2d(64*2),
            nn.ReLU()
        )
        self.y_input_layer = nn.Sequential(
            nn.ConvTranspose2d(10, 64*2, 3, stride=1, padding=0),
            nn.BatchNorm2d(64*2),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            # (64*4) * 3 * 3
            nn.ConvTranspose2d(64*4, 64*2, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),

            # (64*2) * 7 * 7
            nn.ConvTranspose2d(64*2, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),  
            
            # (64) * 14 * 14
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.initialize_weights()

    def forward(self, x, y):
        # x dim : batch * 100 * 1 * 1
        x = self.x_input_layer(x)
        # y dim : batch * 10
        y = y.reshape([y.shape[0], -1, 1, 1])
        # y dim : batch * 100 * 1 * 1
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
        # (1) * 28 * 28
        self.x_input_layer = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        # (10) * 28 * 28
        self.y_input_layer = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.layers = nn.Sequential(
            # (64) * 14 * 14
            nn.Conv2d(64, 64*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2),

            # (64*2) * 7 * 7
            nn.Conv2d(64*2, 64*4, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2),

            # (64*4) * 3 * 3
            nn.Conv2d(64*4, 1, kernel_size=3, stride=1, padding=0, bias=False),
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