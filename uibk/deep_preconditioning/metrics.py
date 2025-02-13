"""A collection of metrics to be used in the project."""

from typing import TYPE_CHECKING

import torch

from uibk.deep_preconditioning.utils import sparse_matvec_mul

if TYPE_CHECKING:
    from spconv.pytorch import SparseConvTensor


def frobenius_loss(
    lower_triangular: "SparseConvTensor", solution: torch.Tensor, right_hand_side: torch.Tensor
) -> torch.Tensor:
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

    return torch.linalg.vector_norm(interim - right_hand_side, ord=2, dim=1).sum()


def inverse_loss(systems_tril: "SparseConvTensor", preconditioners_tril: "SparseConvTensor") -> torch.Tensor:
    """Compute how well the preconditioner approximates the matrix inverse.

    Args:
        systems_tril: Lower triangular matrices as `spconv` tensors.
        preconditioners_tril: Lower triangular matrices as `spconv` tensors.

    Returns:
        The inverse loss on the batch.
    """
    preconditioners = preconditioners_tril.dense()[:, 0]
    preconditioners = torch.matmul(preconditioners, preconditioners.transpose(-1, -2))

    systems = systems_tril.dense()[:, 0]
    systems += torch.tril(systems, -1).transpose(-1, -2)

    preconditioned_systems = torch.matmul(preconditioners, systems)

    identity = (
        torch.eye(systems.shape[1]).unsqueeze(0).expand((systems.shape[0], -1, -1)).to(preconditioned_systems.device)
    )
    return torch.linalg.matrix_norm(preconditioned_systems - identity).mean()


def condition_loss(systems_tril: "SparseConvTensor", preconditioners_tril: "SparseConvTensor") -> torch.Tensor:
    """Compute the condition number loss.

    Args:
        systems_tril: Lower triangular matrices as `spconv` tensors.
        preconditioners_tril: Lower triangular matrices as `spconv` tensors.

    Returns:
        The average condition number of the batch.
    """
    preconditioners = preconditioners_tril.dense()[:, 0]
    preconditioners = torch.matmul(preconditioners, preconditioners.transpose(-1, -2))

    systems = systems_tril.dense()[:, 0]
    systems += torch.tril(systems, -1).transpose(-1, -2)

    preconditioned_systems = torch.matmul(preconditioners, systems)

    sigmas = torch.linalg.svdvals(preconditioned_systems)

    return (sigmas.max(dim=1)[0] / sigmas.min(dim=1)[0]).mean()
