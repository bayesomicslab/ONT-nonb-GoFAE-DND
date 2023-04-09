import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_P1(nn.Module):
    def __init__(self, n_inputs=2, out_dim=256, h_dim=512, Stiefel=False):
        super(Encoder_P1, self).__init__()

        self.n_inputs = n_inputs
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.Stiefel = Stiefel

        self.enc = nn.Sequential(
            nn.Conv1d(self.n_inputs, self.h_dim, 5, 1, 2),
            nn.ELU(inplace=True),
            nn.Conv1d(self.h_dim, self.h_dim, 5, 2, 1),
            nn.ELU(inplace=True),
            nn.Flatten(),
        )

        if not self.Stiefel:
            self.enc_out = nn.Sequential(nn.Linear(self.h_dim * 24, self.out_dim, bias=False))  # 3200

    def forward(self, x):

        x = self.enc(x)

        if not self.Stiefel:
            x = self.enc_out(x)

        return x


class Encoder_P2(nn.Module):
    def __init__(self, in_dim=256, z_dim=64):
        super(Encoder_P2, self).__init__()

        self.in_dim = in_dim
        self.z_dim = z_dim

        # this will be tied
        self.fc = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.z_dim, self.in_dim)).t())

    def forward(self, x):
        x = F.linear(x, self.fc.t())
        return x
