"""Define convolutional neural network architecture for preconditioning.

Classes:
    PreconditionerNet: CNN returns lower triangular matrices for preconditioning.
"""

import spconv.pytorch as spconv
import torch
from torch import nn


class PreconditionerNet(nn.Module):
    """CNN returns preconditioner for conjugate gradient solver."""

    def __init__(self) -> None:
        """Initialize the network architecture."""
        super().__init__()

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

    def forward(self, input_: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Return the `L` part of the `L @ L.T` preconditioner for the conjugate gradient solver.

        Args:
            input_: Sparse batch tensor representing the linear system.

        Returns:
            Sparse batch tensor of lower triangular matrices.
        """
        interim = self.layers(input_)

        (filter, ) = torch.where(interim.indices[:, 1] < interim.indices[:, 2])  # (batch, row, col)
        interim.features[filter] = 0  # make the matrix lower triangular

        # TODO: Check diagonal, maybe enforce positive values?

        return interim
