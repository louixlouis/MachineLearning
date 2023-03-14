import torch
import torch.nn as nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder_layer = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.2)
        )

        # output layer
        self.mu_layer = nn.Sequential(
            nn.Linear(256, 2),
            # nn.ReLU()
        )
        self.log_var_layer = nn.Sequential(
            nn.Linear(256, 2),
            # nn.ReLU()
        )

        self.decoder_layer = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.2)
        )

        # output layer
        self.decoder_fc = nn.Sequential(
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def encoder(self, x):
        out = self.encoder_layer(x)
        mu = self.mu_layer(out)
        log_var = self.log_var_layer(out)
        return mu, log_var
    
    def decoder(self, z):
        out = self.decoder_layer(z)
        out = self.decoder_fc(out)
        return out
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out = self.decoder(z)
        return out