"""Define convolutional neural network architecture for preconditioning."""
import spconv
import torch
from torch import nn


class PrecondNet(nn.Module):
    """CNN returns preconditioner for conjugate gradient solver."""

    def __init__(self):
        super(PrecondNet, self).__init__()
        self.layers = spconv.SparseSequential(
            spconv.SparseConv2d(1, 64, 1),
            nn.PReLU(),
            spconv.SparseConv2d(64, 256, 2, padding=(1, 0)),
            nn.PReLU(),
            spconv.SparseConv2d(256, 512, 2, padding=(1, 0)),
            nn.PReLU(),
            spconv.SparseConv2d(512, 256, 2, padding=(0, 1)),
            nn.PReLU(),
            spconv.SparseConv2d(256, 64, 2, padding=(0, 1)),
            nn.PReLU(),
            spconv.SparseConv2d(64, 1, 1),
        )

    def forward(self, x):
        x = self.layers(x).dense().squeeze()
        L = torch.tril(x, diagonal=-1)
        D = nn.functional.threshold(torch.diag(x), 1e-3, 1e-3)
        x = L + torch.diag(D)
        return x.mm(x.transpose(-2, -1))
