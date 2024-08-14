"""Test the untility functions in the module."""

from pathlib import Path

import spconv.pytorch as spconv
import torch

import uibk.deep_preconditioning.utils as utils


def spconv_tensor_fixture():
    """Fixture to create a `SparseConvTensor` for testing."""
    indices = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [0, 2, 2],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 2, 1],
        ]).int()
    features = torch.tensor([[1, 2, 3, 4, 5, 2, 3, 1, 4, 5]]).T.float()
    return spconv.SparseConvTensor(features, indices, spatial_shape=[3, 3], batch_size=2)


def vector_batch_fixture():
    """Fixture to create a batch of vectors."""
    return torch.tensor([[1, 2, 3], [1, -1, 1]]).float()


def test_sparse_matvec_mul(spconv_tensor, vector_batch):
    """Test the `sparse_matvec_mul` function."""
    result = utils.sparse_matvec_mul(spconv_tensor, vector_batch, transpose=False)

    expected_output = torch.tensor([[5, 11, 15], [1, -3, -5]]).float()

    assert torch.allclose(result, expected_output), f"Expected {expected_output}, but got {result}"


def main():
    """Run the tests."""
    spconv_tensor = spconv_tensor_fixture()
    vector_batch = vector_batch_fixture()

    test_sparse_matvec_mul(spconv_tensor, vector_batch)

    print(f"{Path(__file__).name} all passed")


if __name__ == "__main__":
    main()
