from torch import nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, n_features, n_latent):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, 128),
            nn.ReLU(),
            nn.Linear(128, n_features)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
