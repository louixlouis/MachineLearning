import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder_layer = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 3*3*32),
            nn.ReLU()
        )
        self.decoder_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        out = self.encoder_layer(x)
        latent_vec = self.encoder_fc(out.view(out.shape[0], -1))
        out = self.decoder_fc(latent_vec)
        reconstructed = self.decoder_layer(out.view(out.shape[0], 32, 3, 3))
        return latent_vec, reconstructed

    def decoder(self, latent_vec):
        out = self.decoder_fc(latent_vec)
        reconstructed = self.decoder_layer(out.view(out.shape[0], 32, 3, 3))
        return reconstructed
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)