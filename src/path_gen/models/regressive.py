import torch


class RegressiveLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.LSTM(hidden_size=256, input_size=256, num_layers=1)
        self.decoder = torch.nn.LSTM(hidden_size=256, output_size=256, num_layers=1)

    def forward(self, x):
        out, h0, c0 = self.encoder(x)
        _, reconstructed_x = self.decoder(x, (h0, c0))

        return reconstructed_x
