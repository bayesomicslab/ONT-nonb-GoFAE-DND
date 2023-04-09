import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, n_inputs=64, n_outputs=50, h_dim=512):
        super(Decoder, self).__init__()

        self.n_inputs = n_inputs
        self.h_dim = h_dim
        self.n_outputs = n_outputs

        self.dec_lin0 = nn.Sequential(nn.Linear(self.n_inputs, self.h_dim * 24),
                                      nn.ELU(inplace=True), )  # consider removing ReLU

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(self.h_dim, self.h_dim, 5, 2, 1),  # 32 x 32
            nn.ELU(inplace=True),
            nn.ConvTranspose1d(self.h_dim, self.h_dim, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.ConvTranspose1d(self.h_dim, self.n_outputs, 4, 1, 1),
        )

    def forward(self, x):
        x = self.dec_lin0(x)
        x = x.view(-1, self.h_dim, 24)
        x = self.deconv1(x)
        return x