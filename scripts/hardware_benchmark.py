"""Test the maximum size of `spconv` tensor that still fit on GPU."""

import gc

import dvc.api
import numpy as np
import torch
from scipy.sparse import diags, identity, kron
from spconv.pytorch import SparseConvTensor

import uibk.deep_preconditioning.model as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_poisson_matrix(size: int):
    """Generates the sparse matrix for the 2D Poisson equation on an N x N grid.

    Args:
        size: Grid size (number of points in one dimension).

    Returns:
        Sparse matrix in COO format of shape (N^2, N^2).
    """
    # 1D Laplacian matrix (tridiagonal)
    main_diag = 4 * np.ones(size)
    off_diag = -1 * np.ones(size - 1)
    T = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], shape=(size, size))

    # Identity matrix
    I = identity(size)

    # 2D Laplacian using Kronecker products
    A = (
        kron(I, T)
        + kron(diags([1], [1], shape=(size, size)), -identity(size))
        + kron(diags([1], [-1], shape=(size, size)), -identity(size))
    )

    return A.tocoo()


def to_spconv_tensor(sparse_matrix):
    """Convert a scipy sparse COO matrix to a SparseConvTensor."""
    # Convert to PyTorch sparse COO
    indices = torch.tensor(np.vstack((sparse_matrix.row, sparse_matrix.col)), dtype=torch.int32)
    values = torch.tensor(sparse_matrix.data, dtype=torch.float32).unsqueeze(-1)
    spatial_shape = 2 * [indices.max().item() - 1]
    batch_size = 1

    # Need to add a dummy feature dimension and move to GPU
    features = values.reshape(-1, 1).to(DEVICE)
    indices = indices.t().contiguous()
    indices = torch.cat((torch.zeros(indices.size(0), 1), indices), dim=1).int().to(DEVICE)
    return SparseConvTensor(features, indices, spatial_shape, batch_size)


def main() -> None:
    """The main harware benchmark loop."""
    params = dvc.api.params_show()
    print(f"{DEVICE} available for testing.")

    for power in np.arange(8, 16):
        model = getattr(models, params["model"])(params["channels"])
        model.to(DEVICE)

        size = np.power(2, power).astype(int)
        try:
            print(f"Trying size 2^{power} = {2**power}")
            A = generate_poisson_matrix(size)
            spconv_tensor = to_spconv_tensor(A, shape=[size * size])
            print(f"Success for tensor shape {spconv_tensor.spatial_shape}\n")
        except Exception as e:
            print(f"Failed at size 2^{power}")
            print("Error:", e)
            break

        del spconv_tensor
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
