"""Implement model training methods and the loop."""

from pathlib import Path
from typing import TYPE_CHECKING

import dvc.api
import numpy as np
import torch
from dvclive import Live
from scipy.sparse import csr_matrix
from torch.utils.data.dataset import random_split

from uibk.deep_preconditioning.data_set import SludgePatternDataSet
from uibk.deep_preconditioning.metrics import inverse_loss
from uibk.deep_preconditioning.model import PreconditionerNet
from uibk.deep_preconditioning.utils import benchmark_cg

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import Dataset, Subset


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
    running_loss = 0

    for index in range(len(data_set)):
        systems_tril, _, _, _ = data_set[index]
        preconditioners_tril = model(systems_tril)

        optimizer.zero_grad()
        loss = inverse_loss(systems_tril, preconditioners_tril)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(data_set)


@torch.no_grad()
def _validate(model: "nn.Module", data_set: "Dataset | Subset") -> tuple[float, ...]:
    """Test the model on the validation data.

    Args:
        model: The model to test.
        data_set: The validation data set.

    Returns:
        The validation loss and metrics of the preconditioned systems.
    """
    model.eval()

    val_losses = list()
    durations = list()
    iterations = list()

    for index in range(len(data_set)):
        systems_tril, _, right_hand_sides, original_sizes = data_set[index]
        preconditioners_tril = model(systems_tril)

        val_losses.append(inverse_loss(systems_tril, preconditioners_tril).item())

        for batch_index in range(systems_tril.batch_size):
            original_size = original_sizes[batch_index]

            system = systems_tril.dense()[batch_index, 0, :original_size, :original_size]
            system += torch.tril(system, -1).transpose(-1, -2)
            system = system.cpu().numpy()

            right_hand_side = right_hand_sides[batch_index, :original_size].squeeze().cpu().numpy()

            preconditioner = preconditioners_tril.dense()[batch_index, 0, :original_size, :original_size]
            preconditioner = torch.matmul(preconditioner, preconditioner.transpose(-1, -2))
            preconditioner = preconditioner.cpu().numpy()

            preconditioner = csr_matrix(preconditioner)
            duration, n_iterations = benchmark_cg(
                system,
                right_hand_side,
                preconditioner=preconditioner,
            )
            durations.append(duration)
            iterations.append(n_iterations)

    return np.mean(val_losses).item(), np.mean(durations).item(), np.mean(iterations).item()


class EarlyStopping():
    """Stop the training when no more significant improvement."""

    def __init__(self, patience: int) -> None:
        """Initialize the early stopping hyperparameters.

        Attributes:
            patience: Steps with no improvement after which training will be stopped.
            local_min: The lowest validation loss cached.
            counter: Epochs with nondecreasing validation loss.
        """
        self.patience = patience
        self.local_min = float("inf")
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        """Check change of validation loss and update counter."""
        if val_loss > self.local_min:
            self.counter += 1
        else:
            self.local_min = val_loss
            self.counter = 0

        return self.counter >= self.patience


def main() -> None:
    """Run the main model training pipeline."""
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    torch.manual_seed(69)

    params = dvc.api.params_show()

    data_set = SludgePatternDataSet(stage="train", batch_size=params["batch_size"], shuffle=True)
    train_data, val_data = random_split(data_set, lengths=[0.95, 0.05])

    model = PreconditionerNet(params["channels"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    early_stopping = EarlyStopping(patience=params["patience"])
    best_val_loss = float("inf")

    live = Live(  # init logger
        dir=str(Path("assets/dvclive/")),
        report="html",
        save_dvc_exp=True,
        dvcyaml=False,
    )

    while True:
        train_loss = _train_single_epoch(model, train_data, optimizer)
        live.log_metric("train/loss/log_inverse", np.log(train_loss))

        val_loss, durations, iterations = _validate(model, val_data)
        live.log_metric("val/loss/log_inverse", np.log(val_loss))
        live.log_metric("val/metric/durations", durations)
        live.log_metric("val/metric/iterations", iterations)

        if early_stopping(val_loss):
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        torch.save(model.state_dict(), Path(f"assets/checkpoints/{model.__class__.__name__}.pt"))

        live.next_step()

    live.end()


if __name__ == "__main__":
    main()
