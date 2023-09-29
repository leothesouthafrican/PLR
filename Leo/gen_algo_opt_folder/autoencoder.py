import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=2):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # you can also use Tanh or other activations depending on the nature of your data.
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(model, dataset, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for data in dataset:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(data)
            loss = criterion(outputs, data)
            
            # Backward
            loss.backward()
            
            # Optimize
            optimizer.step()

    return model

# To get the encoded representation
def get_encoded_representation(model, data):
    return model.encoder(data).detach().numpy()
