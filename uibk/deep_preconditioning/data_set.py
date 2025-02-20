"""A collection of PyTorch data sets from sparse symmetric positive-definite problems.

Classes:
    SludgePatternDataSet: A collection of linear Poisson problems from CFD simulations.
    StAnDataSet: A large collection of solved linear static analysis problems on frame structures.
    RandomSPDDataSet: Random symmetric positive-definite matrices data set.
"""

import random
from pathlib import Path

import kaggle
import numpy as np
import spconv.pytorch as spconv
import torch
from scipy.sparse import coo_matrix, load_npz, save_npz
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
                self.folders = self._folders[: len(self._folders) * 80 // 100]
            case "test":
                self.folders = self._folders[len(self._folders) * 80 // 100 :]
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
            original_sizes += (original_size[0],)
            difference = self.dof_max - original_size[0]

            # filter lower triangular part because of symmetry
            (filter,) = np.where(rows >= columns)
            rows = rows[filter]
            columns = columns[filter]
            values = values[filter]
            # add trivial equations for maximum degrees of freedom
            rows = np.append(rows, np.arange(original_size[0], self.dof_max))
            columns = np.append(columns, np.arange(original_size[0], self.dof_max))
            values = np.append(values, np.ones((difference,)))

            solution = np.loadtxt(case_folder / "solution.csv")
            right_hand_side = np.loadtxt(case_folder / "right_hand_side.csv")

            batch["features"].append(np.expand_dims(values, axis=-1))
            batch["indices"].append(
                np.column_stack(
                    (np.full(len(values), batch_index), rows, columns),
                )
            )
            batch["solutions"].append(
                np.expand_dims(
                    np.pad(solution, (0, difference), "constant", constant_values=1),
                    axis=0,
                )
            )
            batch["right_hand_sides"].append(
                np.expand_dims(
                    np.pad(right_hand_side, (0, difference), "constant", constant_values=1),
                    axis=0,
                )
            )

        features = torch.from_numpy(np.vstack(batch["features"])).float().to(self.device)
        indices = torch.from_numpy(np.vstack(batch["indices"])).int().to(self.device)
        lower_triangular_systems = spconv.SparseConvTensor(
            features, indices, [self.dof_max, self.dof_max], self.batch_size
        )

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
                self.files[index * self.batch_size + batch_index]
            ).values()
            original_sizes += solution.shape
            difference = self.dof_max - len(solution)

            # filter lower triangular part because of symmetry
            (filter,) = np.where(indices[0] >= indices[1])
            indices = indices[:, filter]
            values = values[filter]

            batch["features"].append(np.expand_dims(values, axis=-1))
            batch["indices"].append(np.concatenate((np.full((len(values), 1), batch_index), indices.T), axis=1))
            batch["solutions"].append(
                np.expand_dims(
                    np.pad(solution, (0, difference)),
                    axis=0,
                )
            )
            batch["right_hand_sides"].append(
                np.expand_dims(
                    np.pad(right_hand_side, (0, difference)),
                    axis=0,
                )
            )

        features = torch.from_numpy(np.vstack(batch["features"])).float().to(self.device)
        indices = torch.from_numpy(np.vstack(batch["indices"])).int().to(self.device)
        systems_tril = spconv.SparseConvTensor(features, indices, [self.dof_max, self.dof_max], self.batch_size)

        solutions = torch.from_numpy(np.vstack(batch["solutions"])).float().to(self.device)
        right_hand_sides = torch.from_numpy(np.vstack(batch["right_hand_sides"])).float().to(self.device)

        return systems_tril, solutions, right_hand_sides, original_sizes


class RandomSPDDataSet(Dataset):
    """Random symmetric positive-definite matrices data set."""

    def __init__(
        self, stage: str, dof: int, batch_size: int, sparsity: float = 0.99, length: int = 1000, shuffle: bool = True
    ) -> None:
        """Initialize the data set.

        Args:
            stage: One of "train" or "test" in 80/20 split.
            dof: Degrees of freedom where (dof, dof) is the size of each matrix.
            batch_size: Number of samples per batch.
            sparsity: Percentage in (0, 1] indicating how many off-diagonal elements are zero.
            length: Number of total samples.
            shuffle: Whether to shuffle the data.
        """
        assert torch.cuda.is_available(), "CUDA is mandatory but not available"
        self.device = torch.device("cuda")

        assert 0 < sparsity <= 1, f"`sparsity` must be in (0, 1] but got {sparsity}"

        self.dof = dof
        self.sparsity = sparsity
        self.batch_size = batch_size
        self.length = length

        self.save_dir = ROOT / "random_spd"

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
            self._generate_data_set()

        match stage:
            case "train":
                self.files = list(self.save_dir.glob("*.npz"))[: length * 80 // 100]
            case "test":
                self.files = list(self.save_dir.glob("*.npz"))[length * 80 // 100 :]
            case _:
                raise AssertionError(f"Invalid stage {stage}")
        if shuffle:
            random.shuffle(self.files)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.files) // self.batch_size

    def __getitem__(self, index: int) -> tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
        """Return a single batch of random SPD matrices.

        The tensor format is as required in the `traveller59/spconv` package.
        """
        batch = dict(features=list(), indices=list(), solutions=list(), right_hand_sides=list())
        original_sizes = tuple()

        for batch_index in range(self.batch_size):
            file = self.files[index * self.batch_size + batch_index]

            rows, columns, _, _, values = np.load(file).values()
            matrix = load_npz(file)
            original_sizes += (self.dof,)

            # filter lower triangular part because of symmetry
            (filter,) = np.where(rows >= columns)
            rows = rows[filter]
            columns = columns[filter]
            values = values[filter]

            solution = np.ones((self.dof,))
            right_hand_side = matrix @ solution

            batch["features"].append(np.expand_dims(values, axis=-1))
            batch["indices"].append(
                np.column_stack(
                    (np.full(len(values), batch_index), rows, columns),
                )
            )
            batch["solutions"].append(np.expand_dims(solution, axis=0))
            batch["right_hand_sides"].append(np.expand_dims(right_hand_side, axis=0))

        features = torch.from_numpy(np.vstack(batch["features"])).float().to(self.device)
        indices = torch.from_numpy(np.vstack(batch["indices"])).int().to(self.device)
        lower_triangular_systems = spconv.SparseConvTensor(features, indices, [self.dof, self.dof], self.batch_size)

        solutions = torch.from_numpy(np.vstack(batch["solutions"])).float().to(self.device)
        right_hand_sides = torch.from_numpy(np.vstack(batch["right_hand_sides"])).float().to(self.device)

        return lower_triangular_systems, solutions, right_hand_sides, original_sizes

    def _generate_random_spd_matrix(self) -> np.ndarray:
        """Generate a single random SPD matrix with a given non-zero pattern.

        For this synthetic data set, see also:

        HÄUSNER, Paul; ÖKTEM, Ozan; SJÖLUND, Jens. Neural incomplete factorization: learning preconditioners for the
        conjugate gradient method. arXiv preprint arXiv:2305.16368, 2023.

        https://arxiv.org/pdf/2305.16368
        """
        row_indices, col_indices = np.tril_indices(n=self.dof, k=-1)
        sample_indices = random.sample(range(len(row_indices)), k=int((1 - self.sparsity) * len(row_indices)))

        interim = np.zeros((self.dof, self.dof), dtype=np.float32)

        rng = np.random.default_rng()
        for sample_index in sample_indices:
            interim[row_indices[sample_index], col_indices[sample_index]] = rng.standard_normal()

        alpha = 1e-3
        return interim @ interim.T + alpha * np.eye(self.dof)

    def _generate_data_set(self) -> None:
        """Generate the data set."""
        for index in tqdm(iterable=list(range(self.length)), desc="Generating random SPD matrices", unit="matrices"):
            matrix = coo_matrix(self._generate_random_spd_matrix())
            save_npz(self.save_dir / f"{index:04}.npz", matrix, compressed=False)
