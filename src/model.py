import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.
    """

    def __init__(self, feat_dim, hidden_dim=256, layers=2):
        super().__init__()

        self.encoder = nn.LSTM(feat_dim, hidden_dim, layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, feat_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)

        dec_input = torch.zeros_like(x)
        dec_out, _ = self.decoder(dec_input, (h, c))

        out = self.fc(dec_out)
        return out
