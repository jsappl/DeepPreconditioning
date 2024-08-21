"""Define convolutional neural network architecture for preconditioning.

Classes:
    PreconditionerNet: CNN returns lower triangular matrices for preconditioning.
"""

import spconv.pytorch as spconv
import torch
from torch import nn


class PreconditionerNet(nn.Module):
    """Fully convolutional network mapping matrices to lower triangular matrices."""

    def __init__(self, channels: list[int]) -> None:
        """Initialize the network architecture.

        Args:
            channels: Even (mandatory because padding) number of channels in all layers.
        """
        super().__init__()

        assert len(channels) % 2

        # input layer
        self.layers = spconv.SparseSequential(
            spconv.SparseConv2d(channels[0], channels[1], 1),
            nn.PReLU(),
        )

        # hidden layers
        for index, (in_channels, out_channels) in enumerate(zip(channels[1:-2], channels[2:-1], strict=True)):
            padding = (1, 0) if index < (len(channels) - 2) // 2 else (0, 1)

            self.layers.add(spconv.SparseConv2d(in_channels, out_channels, 2, padding=padding))
            self.layers.add(nn.PReLU())

        # output layer
        self.layers.add(spconv.SparseConv2d(channels[-2], channels[-1], 1))

    def forward(self, input_: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Return the `L` part of the `L @ L.T` preconditioner for the conjugate gradient solver.

        Args:
            input_: Sparse batch tensor representing the linear system.

        Returns:
            Sparse batch tensor of lower triangular matrices.
        """
        interim = self.layers(input_)

        (filter, ) = torch.where(interim.indices[:, 1] < interim.indices[:, 2])  # (batch, row, col)
        interim.features[filter] *= 0  # make the matrix lower triangular

        (filter, ) = torch.where(interim.indices[:, 1] == interim.indices[:, 2])
        interim.features[filter] = nn.functional.softplus(interim.features[filter])  # enforce positive diagonal

        return interim
