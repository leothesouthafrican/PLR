# Autoencoder.py

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, depth=3):
        super(Autoencoder, self).__init__()
        assert depth >= 1, "Depth must be at least 1"

        # Encoder layers
        self.encoder_layers = nn.ModuleList([nn.Linear(input_size, input_size // 2)])
        
        current_input_size = input_size // 2
        for i in range(depth - 1):
            next_input_size = current_input_size // 2
            self.encoder_layers.append(nn.Linear(current_input_size, next_input_size))
            current_input_size = next_input_size

        # Decoder layers
        self.decoder_layers = nn.ModuleList([nn.Linear(input_size // (2**depth), input_size // (2**(depth-1)))])
        for i in range(depth - 1, 0, -1):
            self.decoder_layers.append(nn.Linear(input_size // (2**i), input_size // (2**(i-1))))

        # Print layer structures
        print("Encoder layers:")
        for layer in self.encoder_layers:
            print(layer)
            
        print("\nDecoder layers:")
        for layer in self.decoder_layers:
            print(layer)

        self.relu = nn.ReLU()

    def encode(self, x):
        print("Encoding...")
        for i, layer in enumerate(self.encoder_layers):
            x = self.relu(layer(x))
            print(f"Encoder Layer {i+1}: {x.shape}")
        return x

    def decode(self, x):
        print("Decoding...")
        for i, layer in enumerate(self.decoder_layers):
            x = self.relu(layer(x))
            print(f"Decoder Layer {i+1}: {x.shape}")
        return x

    def forward(self, x):
        print(f"Input: {x.shape}")
        x = self.encode(x)
        x = self.decode(x)
        print(f"Output: {x.shape}")
        return x
