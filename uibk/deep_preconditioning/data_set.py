"""A collection of PyTorch data sets from sparse symmetric positive-definite problems.

Classes:
    StAnDataSet: A large collection of solved linear static analysis problems on frame structures.
"""

from pathlib import Path

import numpy as np
import spconv.pytorch as spconv
import torch
from torch.utils.data import Dataset

ROOT: Path = Path("./assets/data/raw/")
DOF_MAX: int = 5166  # https://www.kaggle.com/datasets/zurutech/stand-small-problems


class StAnDataSet(Dataset):
    """A large collection of solved linear static analysis problems on frame structures.

    See also https://www.kaggle.com/datasets/zurutech/stand-small-problems.
    """

    def __init__(self, stage: str, batch_size: int, root: Path = ROOT) -> None:
        """Initialize the data set.

        Args:
            stage: One of "train" or "test".
            batch_size: Number of samples per batch.
            root: Path to the data directory.

        Raises:
            An `AssertionError` if `stage` is neither "train" nor "test" or if CUDA is not available.
        """
        assert stage in ["train", "test"], f"Invalid stage {stage}"
        self.files = list(root.glob(f"stand_small_{stage}/*.npz"))

        self.batch_size = batch_size

        assert torch.cuda.is_available(), "CUDA is mandatory but not available"
        self.device = torch.device("cuda")

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.files) // self.batch_size

    def __getitem__(self, index: int) -> tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor]:
        """Return a single batch of linear system data.

        The tensor format is as required in the `traveller59/spconv` package. The matrices, solutions, and right-hand
        sides are zero-padded to fit the maximum degrees of freedom.
        """
        batch = dict(features=list(), indices=list(), solution=list(), right_hand_side=list())
        for batch_index in range(index * self.batch_size, (index + 1) * self.batch_size):
            indices, values, solutions, right_hand_sides = np.load(self.files[index]).values()
            batch["features"].append(np.expand_dims(values, axis=-1))
            batch["indices"].append(np.concatenate((np.full((len(values), 1), batch_index), indices.T), axis=1))
            batch["solution"].append(np.expand_dims(
                np.pad(solutions, (0, DOF_MAX - len(solutions))),
                axis=0,
            ))
            batch["right_hand_side"].append(
                np.expand_dims(
                    np.pad(right_hand_sides, (0, DOF_MAX - len(right_hand_sides))),
                    axis=0,
                ))

        features = torch.from_numpy(np.vstack(batch["features"])).float().to(self.device)
        indices = torch.from_numpy(np.vstack(batch["indices"])).int().to(self.device)
        matrices = spconv.SparseConvTensor(features, indices, [DOF_MAX, DOF_MAX], self.batch_size)

        solutions = torch.from_numpy(np.vstack(batch["solution"])).float().to(self.device)
        right_hand_sides = torch.from_numpy(np.vstack(batch["right_hand_side"])).float().to(self.device)

        return matrices, solutions, right_hand_sides
