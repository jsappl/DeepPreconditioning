"""A collection of metrics to be used in the project."""

from typing import TYPE_CHECKING

import torch

from uibk.deep_preconditioning.utils import sparse_matvec_mul

if TYPE_CHECKING:
    from spconv.pytorch import SparseConvTensor


def frobenius_loss(
        lower_triangular: "SparseConvTensor", solution: torch.Tensor, right_hand_side: torch.Tensor) -> torch.Tensor:
    """Compute the Frobenius norm of the error.

    See also <https://arxiv.org/pdf/2305.16432>, equation (11).

    Args:
        lower_triangular: Lower triangular matrices as `spconv` tensors.
        solution: Solution tensors.
        right_hand_side: Right-hand side tensors.

    Returns:
        The Frobenius norm of the error on the batch.
    """
    interim = sparse_matvec_mul(lower_triangular, solution, transpose=True)
    interim = sparse_matvec_mul(lower_triangular, interim, transpose=False)

    return (interim - right_hand_side).square().sum()
