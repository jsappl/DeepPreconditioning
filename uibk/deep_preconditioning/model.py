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

        (filter,) = torch.where(interim.indices[:, 1] < interim.indices[:, 2])  # (batch, row, col)
        interim.features[filter] *= 0  # make the matrix lower triangular

        (filter,) = torch.where(interim.indices[:, 1] == interim.indices[:, 2])
        interim.features[filter] = nn.functional.softplus(interim.features[filter])  # enforce positive diagonal

        return interim


class PreconditionerSparseUNet(nn.Module):
    """U-Net inspired architecture for preconditioning."""

    def __init__(self, channels: list[int]) -> None:
        """Initialize the network architecture."""
        super().__init__()

        self.enc1 = spconv.SparseSequential(
            spconv.SubMConv2d(channels[0], channels[1], kernel_size=3, padding=1, indice_key="subm1"),
            nn.LeakyReLU(),
        )
        self.down1 = spconv.SparseSequential(
            spconv.SparseConv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, indice_key="down1"),
            nn.LeakyReLU(),
        )
        self.enc2 = spconv.SparseSequential(
            spconv.SubMConv2d(channels[2], channels[2], kernel_size=3, padding=1, indice_key="subm2"),
            nn.LeakyReLU(),
        )
        self.down2 = spconv.SparseSequential(
            spconv.SparseConv2d(channels[2], channels[3], kernel_size=3, stride=2, padding=1, indice_key="down2"),
            nn.LeakyReLU(),
        )
        self.enc3 = spconv.SparseSequential(
            spconv.SubMConv2d(channels[3], channels[3], kernel_size=3, padding=1, indice_key="subm3"),
            nn.LeakyReLU(),
        )

        self.bottleneck = spconv.SparseSequential(
            spconv.SparseConv2d(channels[3], channels[4], kernel_size=3, stride=2, padding=1, indice_key="bneck"),
            nn.LeakyReLU(),
        )

        self.up2 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(channels[4], channels[3], kernel_size=3, indice_key="bneck"),
            nn.LeakyReLU(),
        )
        self.dec2 = spconv.SparseSequential(
            spconv.SubMConv2d(channels[3], channels[3], kernel_size=3, padding=1, indice_key="subm3"),
            nn.LeakyReLU(),
        )
        self.up1 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(channels[3], channels[2], kernel_size=3, indice_key="down2"),
            nn.LeakyReLU(),
        )
        self.dec1 = spconv.SparseSequential(
            spconv.SubMConv2d(channels[2], channels[2], kernel_size=3, padding=1, indice_key="subm2"),
            nn.LeakyReLU(),
        )
        self.up0 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(channels[2], channels[1], kernel_size=3, indice_key="down1"),
            nn.LeakyReLU(),
        )
        self.dec0 = spconv.SparseSequential(
            spconv.SubMConv2d(channels[1], channels[1], kernel_size=3, padding=1, indice_key="subm1"),
            nn.LeakyReLU(),
        )

        self.out_conv = spconv.SparseSequential(
            spconv.SubMConv2d(channels[1], channels[5], kernel_size=1, padding=0),
        )

    def forward(self, input_: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Return the `L` part of the `L @ L.T` preconditioner for the conjugate gradient solver."""
        # --- Encoder ---
        enc1 = self.enc1(input_)
        down1 = self.down1(enc1)
        enc2 = self.enc2(down1)  # process at R/2
        down2 = self.down2(enc2)  # downsample R/2 -> R/4
        enc3 = self.enc3(down2)  # process at R/4

        # --- Bottleneck ---
        bottleneck = self.bottleneck(enc3)

        # --- Decoder ---
        up2 = self.up2(bottleneck)
        up2 = spconv.functional.sparse_add(up2, enc3)
        dec2 = self.dec2(up2)

        up1 = self.up1(dec2)
        up1 = spconv.functional.sparse_add(up1, enc2)
        dec1 = self.dec1(up1)

        up0 = self.up0(dec1)
        up0 = spconv.functional.sparse_add(up0, enc1)
        dec0 = self.dec0(up0)

        interim = self.out_conv(dec0)

        (filter,) = torch.where(interim.indices[:, 1] < interim.indices[:, 2])  # (batch, row, col)
        interim.features[filter] *= 0  # make the matrix lower triangular

        (filter,) = torch.where(interim.indices[:, 1] == interim.indices[:, 2])
        interim.features[filter] = nn.functional.softplus(interim.features[filter])  # enforce positive diagonal

        return interim
