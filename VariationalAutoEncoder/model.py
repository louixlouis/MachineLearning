import torch
import torch.nn as nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder_layer = nn.Seuqential(
            '''
            In : Batch * 1 * 28 * 28
            Out: Batch * 32 * 26 * 26
            '''
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            '''
            In : Batch * 32 * 26 * 26
            Out: Batch * 64 * 24 * 24
            '''
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            '''
            In : Batch * 64 * 24 * 24
            Out: Batch * 128 * 22 * 22
            '''
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            '''
            In : Batch * 128 * 22 * 22
            Out: Batch * 256 * 20 * 20
            '''
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            '''
            In : Batch * 256 * 20 * 20
            Out: Batch * 512 * 18 * 18
            '''
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(518*18*18,2)
        self.fc_log_var = nn.Linear(518*18*18,2)

        self.decoder_fc = nn.Linear(2, 518*18*18)
        self.decoder_layer = nn.Sequential(
            
        )

    def forward(self, x):
        pass