"""Test the performance of convetional and our preconditioner."""

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from pyamg.aggregation import smoothed_aggregation_solver
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import inv, spilu
from tqdm import tqdm

from uibk.deep_preconditioning.data_set import SludgePatternDataSet
from uibk.deep_preconditioning.model import PreconditionerNet
from uibk.deep_preconditioning.utils import benchmark_cg

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from scipy.sparse import csc_matrix, spmatrix
    from spconv.pytorch import SparseConvTensor
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
        "incomplete_cholesky",
        "algebraic_multigrid",
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

    def _reconstruct_system(self, system_tril: "SparseConvTensor", original_size: int) -> np.ndarray:
        """Reconstruct the linear system from the sparse tensor."""
        assert system_tril.batch_size == 1, "Set batch size to one for testing"

        matrix = system_tril.dense()[0, 0, :original_size, :original_size]
        matrix += torch.tril(matrix, -1).T

        return matrix.cpu().numpy()

    def _construct_vanilla(self, matrix: np.ndarray) -> csr_matrix:
        """Construct the baseline which is no preconditioner."""
        return csr_matrix(np.eye(matrix.shape[0]))

    def _construct_jacobi(self, matrix: np.ndarray) -> csr_matrix:
        """Construct the Jacobi preconditioner."""
        diagonal = matrix.diagonal()
        return csr_matrix(np.diag(1.0 / diagonal))

    def _construct_incomplete_cholesky(self, matrix: np.ndarray) -> "csc_matrix":
        """Construct the incomplete Cholesky preconditioner."""
        size = matrix.shape[0]

        lu_decomposition = spilu(csr_matrix(matrix).tocsc(), fill_factor=1.0, drop_tol=0.0)
        lower = lu_decomposition.L
        diagonal = diags(lu_decomposition.U.diagonal())  # https://is.gd/5PJcTp

        pr = np.zeros((size, size))
        pc = np.zeros((size, size))

        pr[lu_decomposition.perm_r, np.arange(size)] = 1
        pc[np.arange(size), lu_decomposition.perm_c] = 1

        pr = lil_matrix(pr)
        pc = lil_matrix(pc)

        return inv((pr.T * (lower * diagonal * lower.T) * pc.T).tocsc())

    def _construct_algebraic_multigrid(self, matrix: np.ndarray) -> csr_matrix:
        """Construct the algebraic multigrid preconditioner."""
        preconditioner = smoothed_aggregation_solver(matrix).aspreconditioner(cycle="V")
        return csr_matrix(preconditioner.matmat(np.eye(matrix.shape[0], dtype=np.float32)))

    def _construct_learned(self, system_tril: "SparseConvTensor", original_size: int) -> csr_matrix:
        """Construct our preconditioner."""
        preconditioners_tril = self.model(system_tril)
        preconditioner = preconditioners_tril.dense()[0, 0, :original_size, :original_size]
        preconditioner = torch.matmul(preconditioner, preconditioner.transpose(-1, -2))
        preconditioner = preconditioner.detach().cpu().numpy()
        return csr_matrix(preconditioner)

    def _compute_sparsity(self, matrix: "spmatrix") -> float:
        """Compute the sparsity of a matrix."""
        return 100 * matrix.getnnz() / (matrix.shape[0] * matrix.shape[1])

    def _compute_kappa(self, matrix: np.ndarray, preconditioner: "spmatrix") -> float:
        """Compute the condition number."""
        return np.linalg.cond(preconditioner @ matrix)

    def _compute_eigenvalues(self, matrix: np.ndarray, preconditioner: np.ndarray) -> list:
        """Compute the eigenvalues of a matrix."""
        return scipy.linalg.svdvals(preconditioner @ matrix).tolist()

    def run(self) -> None:
        """Run the whole benchmark suite."""
        for index in tqdm(range(len(self.data_set))):
            system_tril, _, right_hand_side, original_size = self.data_set[index]
            matrix = self._reconstruct_system(system_tril, original_size[0])
            right_hand_side = right_hand_side[0, :original_size[0]].squeeze().cpu().numpy()

            if index == 0:
                eigenvalues = dict(vanilla=self._compute_eigenvalues(matrix, np.eye(matrix.shape[0])))

            for name in self.techniques:
                start_time = time.perf_counter()
                if name == "learned":
                    preconditioner = self._construct_learned(system_tril, original_size[0])
                else:
                    preconditioner = getattr(self, f"_construct_{name}")(matrix)
                stop_time = time.perf_counter()
                setup = stop_time - start_time

                density = self._compute_sparsity(preconditioner)
                duration, iteration, info = benchmark_cg(matrix, right_hand_side, preconditioner)
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
                with open(RESULTS_DIRECTORY / "eigenvalues.csv", "w") as file_io:
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

        with open(RESULTS_DIRECTORY / "table.csv", "w") as file_io:
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

    data_set = SludgePatternDataSet(stage="test", batch_size=1, shuffle=False)

    model = PreconditionerNet(params["channels"])
    model.load_state_dict(torch.load(Path("./assets/checkpoints/best.pt")))
    model = model.to(device)

    suite = BenchmarkSuite(data_set, model)
    suite.run()
    suite.dump_csv()


if __name__ == "__main__":
    main()
