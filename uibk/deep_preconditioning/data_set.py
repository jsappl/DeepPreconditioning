"""A collection of PyTorch data sets from sparse symmetric positive-definite problems.

Classes:
    SludgePatternDataSet: A collection of linear Poisson problems from CFD simulations.
    StAnDataSet: A large collection of solved linear static analysis problems on frame structures.
"""

import random
from pathlib import Path

import kaggle
import numpy as np
import spconv.pytorch as spconv
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
        self._folders = sorted(list((root / "sludge_patterns").glob("case_*")))

        match stage:
            case "train":
                self.folders = self._folders[:len(self._folders) * 80 // 100]
            case "test":
                self.folders = self._folders[len(self._folders) * 80 // 100:]
            case _:
                raise AssertionError(f"Invalid stage {stage}")
        if shuffle:
            random.shuffle(self.folders)

        self.batch_size = batch_size
        self.dof_max = self._compute_max_dof()

        assert torch.cuda.is_available(), "CUDA is mandatory but not available"
        self.device = torch.device("cuda")

    def _compute_max_dof(self) -> int:
        """Compute the maximum degrees of freedom in the data set."""
        max_dof = 0

        for folder in self._folders:
            current = np.load(folder / "matrix.npz")["shape"].max().item()
            if current > max_dof:
                max_dof = current

        assert max_dof > 0, "Maximum degrees of freedom is zero"

        return max_dof

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.folders) // self.batch_size

    def __getitem__(self, index: int) -> tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor, tuple[int]]:
        """Return a single batch of linear system data.

        The tensor format is as required in the `traveller59/spconv` package. The matrices, solutions, and right-hand
        sides are zero-padded to fit the maximum degrees of freedom.
        """
        batch = dict(features=list(), indices=list(), solutions=list(), right_hand_sides=list())
        original_sizes = tuple()

        for batch_index in range(self.batch_size):
            case_folder = self.folders[index * self.batch_size + batch_index]

            rows, columns, _, original_size, values = np.load(case_folder / "matrix.npz").values()
            original_sizes += (original_size[0], )
            difference = self.dof_max - original_size[0]

            # filter lower triangular part because of symmetry
            (filter, ) = np.where(rows >= columns)
            rows = rows[filter]
            columns = columns[filter]
            values = values[filter]
            # add trivial equations for maximum degrees of freedom
            rows = np.append(rows, np.arange(original_size[0], self.dof_max))
            columns = np.append(columns, np.arange(original_size[0], self.dof_max))
            values = np.append(values, np.ones((difference, )))

            solution = np.loadtxt(case_folder / "solution.csv")
            right_hand_side = np.loadtxt(case_folder / "right_hand_side.csv")

            batch["features"].append(np.expand_dims(values, axis=-1))
            batch["indices"].append(np.column_stack((np.full(len(values), batch_index), rows, columns), ))
            batch["solutions"].append(
                np.expand_dims(
                    np.pad(solution, (0, difference), "constant", constant_values=1),
                    axis=0,
                ))
            batch["right_hand_sides"].append(
                np.expand_dims(
                    np.pad(right_hand_side, (0, difference), "constant", constant_values=1),
                    axis=0,
                ))

        features = torch.from_numpy(np.vstack(batch["features"])).float().to(self.device)
        indices = torch.from_numpy(np.vstack(batch["indices"])).int().to(self.device)
        lower_triangular_systems = spconv.SparseConvTensor(
            features, indices, [self.dof_max, self.dof_max], self.batch_size)

        solutions = torch.from_numpy(np.vstack(batch["solutions"])).float().to(self.device)
        right_hand_sides = torch.from_numpy(np.vstack(batch["right_hand_sides"])).float().to(self.device)

        return lower_triangular_systems, solutions, right_hand_sides, original_sizes


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
        match stage:
            case "train" | "test":
                self.files = list(root.glob(f"stand_small_{stage}/*.npz"))
            case _:
                raise AssertionError(f"Invalid stage {stage}")
        if shuffle:
            random.shuffle(self.files)
        self.batch_size = batch_size
        self.dof_max = 5166  # https://www.kaggle.com/datasets/zurutech/stand-small-problems

        assert torch.cuda.is_available(), "CUDA is mandatory but not available"
        self.device = torch.device("cuda")

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.files) // self.batch_size

    def __getitem__(self, index: int) -> tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor, tuple[int]]:
        """Return a single batch of linear system data.

        The tensor format is as required in the `traveller59/spconv` package. The matrices, solutions, and right-hand
        sides are zero-padded to fit the maximum degrees of freedom.
        """
        batch = dict(features=list(), indices=list(), solutions=list(), right_hand_sides=list())
        original_sizes = tuple()

        for batch_index in range(self.batch_size):
            indices, values, solution, right_hand_side = np.load(
                self.files[index * self.batch_size + batch_index]).values()
            original_sizes += solution.shape
            difference = self.dof_max - len(solution)

            # filter lower triangular part because of symmetry
            (filter, ) = np.where(indices[0] >= indices[1])
            indices = indices[:, filter]
            values = values[filter]

            batch["features"].append(np.expand_dims(values, axis=-1))
            batch["indices"].append(np.concatenate((np.full((len(values), 1), batch_index), indices.T), axis=1))
            batch["solutions"].append(np.expand_dims(
                np.pad(solution, (0, difference)),
                axis=0,
            ))
            batch["right_hand_sides"].append(np.expand_dims(
                np.pad(right_hand_side, (0, difference)),
                axis=0,
            ))

        features = torch.from_numpy(np.vstack(batch["features"])).float().to(self.device)
        indices = torch.from_numpy(np.vstack(batch["indices"])).int().to(self.device)
        systems_tril = spconv.SparseConvTensor(features, indices, [self.dof_max, self.dof_max], self.batch_size)

        solutions = torch.from_numpy(np.vstack(batch["solutions"])).float().to(self.device)
        right_hand_sides = torch.from_numpy(np.vstack(batch["right_hand_sides"])).float().to(self.device)

        return systems_tril, solutions, right_hand_sides, original_sizes


class RandomSPDDataSet(Dataset):
    """Random symmetric positive-definite matrices data set."""

    def __init__(self, dof: int, num_nonzeros: int, batch_size: int, length: int = 1000) -> None:
        """Initialize the data set.

        Args:
            dof: Degrees of freedom where (dof, dof) is the size of each matrix.
            num_nonzeros: Total number of non-zero entries in each matrix. Must be >= dof to cover diagonal and
                num_nonzeros - dof must be even.
            batch_size: Number of samples per batch.
            length: Number of total samples.
        """
        assert torch.cuda.is_available(), "CUDA is mandatory but not available"
        self.device = torch.device("cuda")

        assert num_nonzeros >= dof, f"`num_nonzeros` must be at least `dof` {dof} for diagonal coverage."
        assert (num_nonzeros - dof) % 2 == 0, "`num_nonzeros - dof` must be even to form symmetric pairs."

        self.dof = dof
        self.num_nonzeros = num_nonzeros
        self.batch_size = batch_size
        self.length = length

        self.indices = list(range(self.length))
        self.save_dir = ROOT / "random_spd"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._generate_data_set()

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.length // self.batch_size

    def __getitem__(self, index: int) -> spconv.SparseConvTensor:
        """Return a single batch of random SPD matrices.

        The tensor format is as required in the `traveller59/spconv` package.
        """
        batch = {"features": [], "indices": []}

        for batch_index in range(self.batch_size):
            index = self.indices[index * self.batch_size + batch_index]
            matrix = np.load(self.save_dir / f"{index:04}.npz")["M"]

            rows, cols = np.where(matrix != 0)
            values = matrix[rows, cols]

            batch["features"].append(np.expand_dims(values, axis=-1))
            batch["indices"].append(np.column_stack((np.full(len(values), batch_index), rows, cols)))

        features = torch.from_numpy(np.vstack(batch["features"])).float().to(self.device)
        indices = torch.from_numpy(np.vstack(batch["indices"])).int().to(self.device)

        return spconv.SparseConvTensor(features, indices, [self.dof, self.dof], self.batch_size)

    def _generate_random_spd_matrix(self) -> np.ndarray:
        """Generate a single random SPD matrix with a given non-zero pattern."""
        # Number of off-diagonal pairs
        off_diag_pairs = (self.num_nonzeros - self.dof) // 2

        # Pick off-diagonal pairs
        chosen_pairs = set()
        while len(chosen_pairs) < off_diag_pairs:
            row_index = np.random.randint(0, self.dof)
            col_index = np.random.randint(0, self.dof)

            if row_index < col_index:
                chosen_pairs.add((row_index, col_index))

        matrix = np.zeros((self.dof, self.dof), dtype=np.float32)

        # Fill diagonal
        for row_index in range(self.dof):
            matrix[row_index, row_index] = np.random.uniform(0.1, 1.0)

            # Fill off-diagonal entries symmetrically
        for (row_index, col_index) in chosen_pairs:
            value = np.random.randn() * 0.1
            matrix[row_index, col_index] = value
            matrix[col_index, row_index] = value

        # Make M diagonally dominant to ensure SPD
        for row_index in range(self.dof):
            off_diag_sum = np.sum(np.abs(matrix[row_index, :])) - abs(matrix[row_index, row_index])
            matrix[row_index, row_index] += off_diag_sum + 1e-1

        return matrix

    def _generate_data_set(self) -> None:
        """Generate the data set."""
        for index in tqdm(iterable=self.indices, desc="Generating random SPD matrices", unit="matrix"):
            matrix = self._generate_random_spd_matrix()
            np.savez(self.save_dir / f"{index:04}.npz", M=matrix)
