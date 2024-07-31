"""A collection of PyTorch data sets from sparse symmetric positive-definite problems.

Classes:
    StAnDataSet: A large collection of solved linear static analysis problems on frame structures.
"""

import random
from pathlib import Path

import kaggle
import numpy as np
import spconv.pytorch as spconv
import torch
from torch.utils.data import Dataset

ROOT: Path = Path("./assets/data/raw/")


class SludgePatternDataSet(Dataset):
    """A collection of linear Poisson problems from CFD."""

    def __init__(self, stage: str, batch_size: int, shuffle: bool = True, root: Path = ROOT) -> None:
        """Initialize the data set.

        Args:
            stage: One of "train" or "test" in 80/20 split.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
            root: Path to the data directory.

        Raises:
            An `AssertionError` if `stage` is neither "train" nor "test" or if CUDA is not available.
        """
        self._files = sorted(list(root.glob("sludge_patterns/*.npz")))

        match stage:
            case "train":
                self.files = self._files[:len(self._files) * 80 // 100]
            case "test":
                self.files = self._files[len(self._files) * 80 // 100:]
            case _:
                raise AssertionError(f"Invalid stage {stage}")
        if shuffle:
            random.shuffle(self.files)

        self.batch_size = batch_size
        self.dof_max = self._compute_max_dof()

        assert torch.cuda.is_available(), "CUDA is mandatory but not available"
        self.device = torch.device("cuda")

    def _compute_max_dof(self) -> int:
        """Compute the maximum degrees of freedom in the data set."""
        max_dof = 0

        for file in self._files:
            current = np.load(file)["shape"].max().item()
            if current > max_dof:
                max_dof = current

        assert max_dof > 0, "Maximum degrees of freedom is zero"

        return max_dof

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.files) // self.batch_size

    def __getitem__(self, index: int) -> tuple[spconv.SparseConvTensor, tuple[int]]:
        """Return a single batch of linear system data.

        The tensor format is as required in the `traveller59/spconv` package. The matrices, solutions, and right-hand
        sides are zero-padded to fit the maximum degrees of freedom.
        """
        batch = dict(features=list(), indices=list())
        original_sizes = tuple()

        for batch_index in range(self.batch_size):
            rows, columns, _, original_size, values = np.load(
                self.files[index * self.batch_size + batch_index]).values()
            original_sizes += (original_size[0], )

            # filter lower triangular part because of symmetry
            (filter, ) = np.where(rows >= columns)
            rows = rows[filter]
            columns = columns[filter]
            values = values[filter]

            batch["features"].append(np.expand_dims(values, axis=-1))
            batch["indices"].append(np.column_stack((np.full(len(values), batch_index), rows, columns), ))

        features = torch.from_numpy(np.vstack(batch["features"])).float().to(self.device)
        indices = torch.from_numpy(np.vstack(batch["indices"])).int().to(self.device)
        lower_triangular_systems = spconv.SparseConvTensor(
            features, indices, [self.dof_max, self.dof_max], self.batch_size)

        return lower_triangular_systems, original_sizes


def download_from_kaggle() -> None:
    """Download the StAn data set from Kaggle."""
    assert Path.home() / ".kaggle/kaggle.json", "Kaggle API key is missing"

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset="zurutech/stand-small-problems", path=ROOT, quiet=False, unzip=False)


class StAnDataSet(Dataset):
    """A large collection of solved linear static analysis problems on frame structures.

    See also https://www.kaggle.com/datasets/zurutech/stand-small-problems.
    """

    def __init__(self, stage: str, batch_size: int, shuffle: bool, root: Path = ROOT) -> None:
        """Initialize the data set.

        Args:
            stage: One of "train" or "test".
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
            root: Path to the data directory.

        Raises:
            An `AssertionError` if `stage` is neither "train" nor "test" or if CUDA is not available.
        """
        assert stage in ["train", "test"], f"Invalid stage {stage}"
        self.files = list(root.glob(f"stand_small_{stage}/*.npz"))
        if shuffle:
            random.shuffle(self.files)

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


if __name__ == "__main__":
    download_from_kaggle()
