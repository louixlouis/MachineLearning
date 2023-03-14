import torch
import torch.nn as nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder_layer_1 = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.encoder_layer_2 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # output layer
        self.mu_layer = nn.Sequential(
            nn.Linear(400, 10),
            nn.ReLU()
        )
        self.log_var_layer = nn.Sequential(
            nn.Linear(400, 10),
            nn.ReLU()
        )

        self.decoder_layer_1 = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # output layer
        self.decoder_fc = nn.Sequential(
            nn.Linear(400, 784),
            nn.Sigmoid()
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
        return eps * std + mu
    
    def encoder(self, x):
        # out = self.encoder_layer_1(x.view(x.shape[0], -1))
        out = self.encoder_layer_1(x)
        out = self.encoder_layer_2(out)
        mu = self.mu_layer(out)
        log_var = self.log_var_layer(out)
        return mu, log_var
    
    def decoder(self, z):
        out = self.decoder_layer_1(z)
        out = self.decoder_layer_2(out)
        out = self.decoder_fc(out)
        return out
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out