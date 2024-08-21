"""A collection of utility functions for the project."""

import time
from typing import TYPE_CHECKING

import torch
from scipy.sparse.linalg import cg

if TYPE_CHECKING:
    from numpy import ndarray
    from scipy.sparse import csr_matrix
    from spconv.pytorch import SparseConvTensor


def sparse_matvec_mul(spconv_batch: "SparseConvTensor", vector_batch: torch.Tensor, transpose: bool) -> torch.Tensor:
    """Perform a sparse matrix-vector multiplication.

    Args:
        spconv_batch: A single batch as an `spconv` tensor.
        vector_batch: A batch of vectors to multiply with the matrix.
        transpose: Whether to transpose the matrix batch.

    Returns:
        The result of the batched matrix-vector multiplication.
    """
    batch_indices = spconv_batch.indices[:, 0]
    row_indices = spconv_batch.indices[:, 2 if transpose else 1]
    column_indices = spconv_batch.indices[:, 1 if transpose else 2]
    output_batch = torch.zeros_like(vector_batch, device=vector_batch.device)

    spconv_batch = spconv_batch.replace_feature(
        spconv_batch.features * vector_batch[batch_indices, column_indices].unsqueeze(-1))
    for batch_index in range(spconv_batch.batch_size):
        filter = torch.where(batch_indices == batch_index)
        output_batch[batch_index] = torch.zeros(vector_batch.shape[-1], device=vector_batch.device).scatter_reduce(
            dim=0,
            index=row_indices[filter].to(torch.int64),
            src=spconv_batch.features[filter].squeeze(),
            reduce="sum",
        )

    return output_batch


def benchmark_cg(matrix: "ndarray", right_hand_side: "ndarray",
                 preconditioner: "ndarray | csr_matrix | None" = None) -> tuple[float, int]:
    """Benchmark the (preconditioned) conjugate gradient method.

    Args:
        matrix: The matrix of the linear system.
        right_hand_side: The right-hand side of the linear system.
        preconditioner: The preconditioner for the conjugate gradient method.

    Returns:
        The duration and number of iterations until convergence.
    """
    iterations = 0

    def _callback(_):
        nonlocal iterations
        iterations += 1

    start_time = time.monotonic_ns()
    cg(
        matrix,
        right_hand_side,
        maxiter=512,
        M=preconditioner,
        callback=_callback,
    )
    duration = (time.monotonic_ns() - start_time) / 1e9  # convert to seconds

    return duration, iterations
