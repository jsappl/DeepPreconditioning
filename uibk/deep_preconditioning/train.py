"""Implement model training methods and the loop."""

from typing import TYPE_CHECKING

import torch
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from uibk.deep_preconditioning.data_set import StAnDataSet
from uibk.deep_preconditioning.metrics import frobenius_loss
from uibk.deep_preconditioning.model import PreconditionerNet

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import Dataset, Subset

BATCH_SIZE: int = 32

def _train_single_epoch(model: "nn.Module", data_set: "Dataset | Subset", optimizer: "Optimizer") -> float:
    """Train the model for a single epoch.

    Args:
        model: The model to train.
        data_set: The training data set.
        optimizer: The optimizer to train the model with.

    Returns:
        The average Frobenius loss on the training data.
    """
    model.train()

    for index in tqdm(range(len(data_set))):
        matrix, solution, right_hand_side = data_set[index]
        lower_triangular = model(matrix)

        optimizer.zero_grad()
        loss = frobenius_loss(lower_triangular, solution, right_hand_side)
        loss.backward()
        optimizer.step()


def main() -> None:
    """Run the main training loop."""
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    model = PreconditionerNet().to(device)

    data_set = StAnDataSet(stage="train", batch_size=BATCH_SIZE, shuffle=True)
    train_data, val_data = random_split(data_set, lengths=[0.95, 0.05])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(10):
        _train_single_epoch(model, train_data, optimizer)


if __name__ == "__main__":
    main()
