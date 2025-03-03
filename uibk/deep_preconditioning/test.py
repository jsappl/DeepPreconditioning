"""Test the performance of convetional and our preconditioner."""

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import dvc.api
import ilupp
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyamg.aggregation import smoothed_aggregation_solver
from scipy.sparse import csr_matrix
from tqdm import tqdm

import uibk.deep_preconditioning.data_set as data_sets
import uibk.deep_preconditioning.model as models
from uibk.deep_preconditioning.cg import preconditioned_conjugate_gradient

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from spconv.pytorch import SparseConvTensor
    from torch import Tensor
    from torch.utils.data import Dataset

RESULTS_DIRECTORY: Path = Path("./assets/results/")


@dataclass
class BenchmarkSuite:
    """Data class that holds the preconditioner benchmark suite.

    Args:
        data_set: The test data set to benchmark on.
        model: Our fully convolutional model to benchmark.
    """

    data_set: "Dataset"
    model: torch.nn.Module
    techniques: tuple[str, ...] = (
        "vanilla",
        "jacobi",
        "incomplete_cholesky",  # unstable
        # "incomplete_lu",  # unstable
        # "algebraic_multigrid",
        "learned",
    )
    kappas = {name: [] for name in techniques}
    densities = {name: [] for name in techniques}
    iterations = {name: [] for name in techniques}
    setups = {name: [] for name in techniques}
    durations = {name: [] for name in techniques}
    totals = {name: [] for name in techniques}
    successes = {name: [] for name in techniques}
    histograms = dict()

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    def _reconstruct_system(self, system_tril: "SparseConvTensor", original_size: int) -> "Tensor":
        """Reconstruct the linear system from the sparse tensor."""
        assert system_tril.batch_size == 1, "Set batch size to one for testing"

        matrix = system_tril.dense()[0, 0, :original_size, :original_size]
        matrix += torch.tril(matrix, -1).T

        return matrix.cpu().to(torch.float64)

    def _construct_vanilla(self, matrix: "Tensor") -> "Tensor":
        """Construct the baseline which is no preconditioner."""
        return torch.eye(matrix.shape[0], dtype=torch.float64).to_sparse_csr()

    def _construct_jacobi(self, matrix: "Tensor") -> "Tensor":
        """Construct the Jacobi preconditioner."""
        data = 1 / matrix.diagonal()
        indices = torch.vstack((torch.arange(matrix.shape[0]), torch.arange(matrix.shape[0])))
        diagonal = torch.sparse_coo_tensor(indices, data, size=matrix.shape, dtype=torch.float64)
        return diagonal.to_sparse_csr()

    def _construct_incomplete_cholesky(self, matrix: "Tensor", fill_in: int = 1, threshold: float = 0.1) -> "Tensor":
        """Construct the incomplete Cholesky preconditioner."""
        if fill_in == 0 and threshold == 0.0:
            icholprec = ilupp.ichol0(csr_matrix(matrix.numpy()))
        else:
            icholprec = ilupp.icholt(csr_matrix(matrix.numpy()), add_fill_in=fill_in, threshold=threshold)

        return torch.from_numpy((icholprec @ icholprec.T).toarray()).to_sparse_csr()

    def _construct_incomplete_lu(self, matrix: "Tensor") -> "Tensor":
        """Construct the incomplete LU preconditioner."""
        l_factor, u_factor = ilupp.ilut(csr_matrix(matrix.numpy()))
        return torch.from_numpy((l_factor @ u_factor).toarray()).to_sparse_csr()

    def _construct_algebraic_multigrid(self, matrix: "Tensor") -> "Tensor":
        """Construct the algebraic multigrid preconditioner."""
        preconditioner = smoothed_aggregation_solver(matrix.numpy()).aspreconditioner(cycle="V")
        return torch.from_numpy(preconditioner.matmat(np.eye(matrix.shape[0], dtype=np.float64))).to_sparse_csr()

    def _construct_learned(self, system_tril: "SparseConvTensor", original_size: int) -> "Tensor":
        """Construct our preconditioner."""
        preconditioners_tril = self.model(system_tril)
        preconditioner = preconditioners_tril.dense()[0, 0, :original_size, :original_size]
        preconditioner = torch.matmul(preconditioner, preconditioner.transpose(-1, -2))
        return preconditioner.detach().cpu().to(torch.float64).to_sparse_csr()

    def _compute_sparsity(self, matrix: "Tensor") -> float:
        """Compute the sparsity of a matrix."""
        return 100 * len(matrix.values()) / (matrix.shape[0] * matrix.shape[1])

    def _compute_kappa(self, matrix: "Tensor", preconditioner: "Tensor") -> float:
        """Compute the condition number."""
        return torch.linalg.cond(preconditioner @ matrix).item()

    def _compute_eigenvalues(self, matrix: "Tensor", preconditioner: "Tensor") -> list:
        """Compute the eigenvalues of a matrix."""
        return torch.linalg.svdvals(preconditioner @ matrix).tolist()

    def run(self) -> None:
        """Run the whole benchmark suite."""
        for index in tqdm(range(len(self.data_set))):
            system_tril, _, right_hand_side, original_size = self.data_set[index]
            matrix = self._reconstruct_system(system_tril, original_size[0])
            right_hand_side = right_hand_side[0, : original_size[0]].squeeze().cpu().to(torch.float64)

            eigenvalues = dict.fromkeys(self.techniques)

            for name in self.techniques:
                if name == "learned":
                    start_time = time.perf_counter()
                    preconditioner = self._construct_learned(system_tril, original_size[0])
                else:
                    start_time = time.perf_counter()
                    preconditioner = getattr(self, f"_construct_{name}")(matrix)
                setup = time.perf_counter() - start_time if name != "vanilla" else 0.0

                density = self._compute_sparsity(preconditioner)
                duration, iteration, info = preconditioned_conjugate_gradient(matrix, right_hand_side, preconditioner)
                kappa = self._compute_kappa(matrix, preconditioner)
                if index == 0:
                    eigenvalues[name] = self._compute_eigenvalues(matrix, preconditioner)

                self.kappas[name].append(kappa)
                self.densities[name].append(density)
                self.iterations[name].append(iteration)
                self.setups[name].append(setup)
                self.durations[name].append(duration)
                self.totals[name].append(setup + duration)
                self.successes[name].append(100 * (1 - info))

            if index == 0:
                with (RESULTS_DIRECTORY / "eigenvalues.csv").open(mode="w") as file_io:
                    writer = csv.writer(file_io)
                    writer.writerow(eigenvalues.keys())
                    writer.writerows(zip(*eigenvalues.values(), strict=True))

    def plot_histograms(self) -> Generator[tuple[str, "Figure"], None, None]:
        """Plot histograms for the durations and iterations."""
        for parameter, label in zip(
            ["durations", "iterations"],
            ["Durations [ms]", "Iterations [-]"],
            strict=True,
        ):
            figure, ax = plt.subplots()

            ax.set_ylabel(label)
            ax.boxplot(
                [getattr(self, parameter)[name] for name in self.durations.keys()],
                notch=True,
                tick_labels=self.techniques,
            )

            yield parameter, figure

    def dump_csv(self) -> None:
        """Dump the durations and iterations to a CSV file.

        Keep in mind that it has to be consumed and rendered using LaTeX later on.
        """
        parameters = ["kappas", "densities", "iterations", "setups", "durations", "totals", "successes"]

        with (RESULTS_DIRECTORY / "table.csv").open(mode="w") as file_io:
            file_io.write("technique," + ",".join(parameters) + "\n")

            for technique in self.techniques:
                line = technique

                for parameter in parameters:
                    line += "," + str(np.mean(getattr(self, parameter)[technique], dtype=float))

                file_io.write(line + "\n")

        with (RESULTS_DIRECTORY / "totals.csv").open(mode="w") as file_io:
            file_io.write(",".join(self.techniques) + "\n")

            for index in range(len(self.totals["vanilla"])):
                line = ",".join([str(self.totals[technique][index]) for technique in self.techniques])
                file_io.write(line + "\n")


def main():
    """Run the main test loop."""
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    torch.manual_seed(69)

    params = dvc.api.params_show()

    data_set = getattr(data_sets, params["data"])(
        stage="test",
        batch_size=1,
        shuffle=False,
    )

    model = getattr(models, params["model"])(params["channels"])
    model.load_state_dict(torch.load(Path("./assets/checkpoints/best.pt")))
    model = model.to(device)

    suite = BenchmarkSuite(data_set, model)
    suite.run()
    suite.dump_csv()


if __name__ == "__main__":
    main()
