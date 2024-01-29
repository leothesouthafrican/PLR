import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, random_seed=42):
        super(Autoencoder, self).__init__()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def reset_weights(self):
        # Reset weights to initial state
        self.apply(self._init_weights)


def train_autoencoder(model, data, learning_rate, epochs):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(data, data), batch_size=32, shuffle=False)

    model.train()
    for epoch in range(epochs):
        for batch_features, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()

    return model

def get_latent_representation(model, data):
    model.eval()
    with torch.no_grad():
        latent_representation = model.encoder(data)
    return latent_representation