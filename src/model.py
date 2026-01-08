import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder.
    Learns normal motion patterns via reconstruction.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )

        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)

        dec_input = torch.zeros_like(x)
        dec_out, _ = self.decoder(dec_input, (h, c))

        out = self.output_layer(dec_out)
        return out
