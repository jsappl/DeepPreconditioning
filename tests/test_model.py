"""Test the implementation of various models."""

from pathlib import Path

import spconv.pytorch as spconv
import torch

import uibk.deep_preconditioning.model as model

SIZE: int = 64
BATCH_SIZE: int = 2


def precondnet_fixture(device):
    """Return an instance of `PreconditionerNet`."""
    return model.PreconditionerNet().to(device)


def spconv_tensor_fixture(device):
    """Fixture to create a `SparseConvTensor` for testing."""
    eye = torch.eye(SIZE, device=device).unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
    return spconv.SparseConvTensor.from_dense(eye.permute(0, 2, 3, 1))


def test_forward(model, input_):
    """Test the forward pass of the PreconditionerNet."""
    lower_triangular = model(input_).dense()

    assert lower_triangular.shape[2:] == torch.Size(input_.spatial_shape)

    for batch_index in range(input_.batch_size):
        assert torch.all(lower_triangular[batch_index, 0].diag() != 0)
        assert torch.all(lower_triangular[batch_index, 0].triu(diagonal=1) == 0)
        assert torch.any(lower_triangular[batch_index, 0].tril(diagonal=-1) != 0)

    preconditioner = lower_triangular.matmul(lower_triangular.transpose(-1, -2)).squeeze()
    assert preconditioner.shape == (BATCH_SIZE, SIZE, SIZE)
    assert torch.all(preconditioner == preconditioner.transpose(-1, -2))

    eigenvalues = torch.linalg.eigvals(preconditioner)
    assert torch.all(eigenvalues.imag == 0)
    assert torch.all(eigenvalues.real > 0)


def main():
    """Run the tests."""
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")

    model = precondnet_fixture(device)
    spconv_tensor = spconv_tensor_fixture(device)

    test_forward(model, spconv_tensor)

    print(f"{Path(__file__).name} all passed")


if __name__ == "__main__":
    main()
