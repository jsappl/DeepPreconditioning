"""Test the performance of convetional and our preconditioner."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import torch
from dvclive import Live
from pyamg.aggregation import smoothed_aggregation_solver
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import inv, spilu

from uibk.deep_preconditioning.data_set import SludgePatternDataSet
from uibk.deep_preconditioning.model import PreconditionerNet
from uibk.deep_preconditioning.utils import benchmark_cg

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from scipy.sparse import csc_matrix
    from spconv.pytorch import SparseConvTensor
    from torch.utils.data import Dataset


@dataclass
class BenchmarkSuite:
    """Data class that holds the preconditioner benchmark suite.

    Args:
        data_set: The test data set to benchmark on.
        model: Our fully convolutional model to benchmark.
    """
    data_set: "Dataset"
    model: torch.nn.Module
    technique: tuple[str, ...] = (
        "vanilla",
        "jacobi",
        "incomplete_cholesky",
        "algebraic_multigrid",
        "learned",
    )
    durations = {name: [] for name in technique}
    iterations = {name: [] for name in technique}
    histograms = dict()

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

    def _construct_learned(self, system_tril: "SparseConvTensor", original_size: int):
        """Construct our preconditioner."""
        preconditioners_tril = self.model(system_tril)
        preconditioner = preconditioners_tril.dense()[0, 0, :original_size, :original_size]
        preconditioner = torch.matmul(preconditioner, preconditioner.transpose(-1, -2))
        preconditioner = preconditioner.detach().cpu().numpy()
        return csr_matrix(preconditioner)

    def run(self) -> None:
        """Run the whole benchmark suite."""
        for index in range(len(self.data_set)):
            system_tril, _, right_hand_side, original_size = self.data_set[index]
            matrix = self._reconstruct_system(system_tril, original_size[0])
            right_hand_side = right_hand_side[0, :original_size[0]].squeeze().cpu().numpy()

            for name in self.technique:
                if name == "learned":
                    continue

                preconditioner = getattr(self, f"_construct_{name}")(matrix)
                duration, iteration = benchmark_cg(matrix, right_hand_side, preconditioner)

                self.durations[name].append(duration)
                self.iterations[name].append(iteration)

            preconditioner = self._construct_learned(system_tril, original_size[0])
            duration, iteration = benchmark_cg(matrix, right_hand_side, preconditioner)

            self.durations["learned"].append(duration)
            self.iterations["learned"].append(iteration)

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
                tick_labels=self.technique,
            )

            yield parameter, figure


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

    live = Live(  # init logger
        dir=str(Path("assets/dvclive/")),
        resume=True,
        report="html",
        save_dvc_exp=False,
        dvcyaml=None,
    )

    suite = BenchmarkSuite(data_set, model)
    suite.run()

    for name, duration in suite.durations.items():
        live.log_metric(f"test/{name}/duration", np.mean(duration, dtype=float))
    for name, iteration in suite.iterations.items():
        live.log_metric(f"test/{name}/iterations", np.mean(iteration, dtype=float))

    for parameter, figure in suite.plot_histograms():
        live.log_image(f"test/histogram/{parameter}.png", figure)

    live.end()


if __name__ == "__main__":
    main()
