import torch
import torch.nn as nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        # self.encoder_layer = nn.Seuqential(
        #     '''
        #     In : Batch * 1 * 28 * 28
        #     Out: Batch * 32 * 26 * 26
        #     '''
        #     nn.Conv2d(1, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(),
        #     '''
        #     In : Batch * 32 * 26 * 26
        #     Out: Batch * 64 * 24 * 24
        #     '''
        #     nn.Conv2d(32, 64, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        #     '''
        #     In : Batch * 64 * 24 * 24
        #     Out: Batch * 128 * 22 * 22
        #     '''
        #     nn.Conv2d(64, 128, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(),
        #     '''
        #     In : Batch * 128 * 22 * 22
        #     Out: Batch * 256 * 20 * 20
        #     '''
        #     nn.Conv2d(128, 256, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(),
        #     '''
        #     In : Batch * 256 * 20 * 20
        #     Out: Batch * 512 * 18 * 18
        #     '''
        #     nn.Conv2d(256, 512, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(),
        # )
        # self.fc_mu = nn.Linear(518*18*18,2)
        # self.fc_log_var = nn.Linear(518*18*18,2)

        # self.decoder_fc = nn.Linear(2, 518*18*18)
        # self.decoder_layer = nn.Sequential(
        #     2, 
        # )
        self.encoder_layer = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            # nn.Linear(128, 4)
            nn.Linear(128, 2)
        )
        # ln(Variance)
        self.fc_log_var = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(2, 128),
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

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps*std + mu
    
    def encoder(self, x):
        out = self.encoder_layer(x)
        mu = self.fc_mu(out.view(out.shape[0], -1))
        log_var = self.fc_log_var(out.view(out.shape[0], -1))
        return mu, log_var
    
    def decoder(self, z):
        out = self.decoder_fc(z)
        out = self.decoder_layer(out.view(out.shape[0], 32, 3, 3))
        return out
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out