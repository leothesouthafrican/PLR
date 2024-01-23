#autoencoder.py

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, first_hidden_dim, depth=3):
        super(Autoencoder, self).__init__()
        assert depth >= 1, "Depth must be at least 1"

        # Encoder layers: Start with the initial hidden layer
        self.encoder_layers = nn.ModuleList([nn.Linear(input_size, first_hidden_dim)])
        
        current_input_size = first_hidden_dim
        for i in range(depth - 1):
            next_input_size = current_input_size // 2
            self.encoder_layers.append(nn.Linear(current_input_size, next_input_size))
            current_input_size = next_input_size

        # Decoder layers: Start the inverse from the last encoder layer's dimension
        self.decoder_layers = nn.ModuleList()
        while current_input_size < first_hidden_dim:
            next_output_size = current_input_size * 2
            self.decoder_layers.append(nn.Linear(current_input_size, next_output_size))
            current_input_size = next_output_size
        
        self.decoder_layers.append(nn.Linear(current_input_size, input_size))

        self.relu = nn.ReLU()

    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.relu(layer(x))
        return x

    def decode(self, x):
        for layer in self.decoder_layers:
            x = self.relu(layer(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


if __name__ == "__main__":
    # Define parameters
    input_size = 144
    first_hidden_dim = 512
    depth = 5
    
    # Create model
    model = Autoencoder(input_size, first_hidden_dim, depth)
    
    # Print model structure
    print(model)
    
    # Test model with random input
    x = torch.randn((5, input_size))  # batch of 5 samples with `input_size` features
    output = model(x)
    
    # Print input and output shapes to confirm the forward pass
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
