"""Train the model."""

import torch
from tqdm import tqdm

from uibk.deep_preconditioning.data_set import StAnDataSet
from uibk.deep_preconditioning.metrics import frobenius_loss
from uibk.deep_preconditioning.model import PreconditionerNet

BATCH_SIZE: int = 32


def _train_single_epoch(model, data_set, optimizer):
    """Train the model for a single epoch."""
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
    data_set = StAnDataSet("train", batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(10):
        _train_single_epoch(model, data_set, optimizer)


if __name__ == "__main__":
    main()
