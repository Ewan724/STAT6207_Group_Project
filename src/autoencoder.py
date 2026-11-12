import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(256,64), dropout=0.2):
        super().__init__()
        h1,h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
