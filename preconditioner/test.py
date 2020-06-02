"""Test preconditioner performance of trained PrecondNet."""
import numpy as np
import torch
from pyamg.aggregation import smoothed_aggregation_solver
from scipy.sparse import csr_matrix, diags, eye, lil_matrix
from scipy.sparse import linalg as sla
from spconv import SparseConvTensor
from torch.utils.tensorboard import SummaryWriter

from config import CFG
from data_loader import init_loaders
from model import PrecondNet
from utils import evaluate


def test(model, writer, device):
    """Test loop for whole test data set."""
    model.eval()
    _, _, test_loader = init_loaders()
    data = np.zeros((len(test_loader), 5, 4))
    for idx, (features, coors, shape, l_matrix) in enumerate(test_loader):
        l_matrix = csr_matrix(l_matrix[0].to_dense().numpy(), dtype=np.float32)
        rhs = np.random.randn(shape[0])

        # Vanilla conjugate gradients without preconditioner as baseline.
        data[idx, 0] = evaluate('vanilla', l_matrix, rhs, eye(shape[0]))

        # Jacobi preconditioner.
        data[idx, 1] = evaluate('jacobi', l_matrix, rhs,
                                diags(1./l_matrix.diagonal()))

        # Incomplete Cholesky preconditioner.
        lu = sla.spilu(l_matrix.tocsc(), fill_factor=1., drop_tol=0.)
        L = lu.L
        D = diags(lu.U.diagonal())  # https://is.gd/5PJcTp
        Pr = np.zeros(l_matrix.shape)
        Pc = np.zeros(l_matrix.shape)
        Pr[lu.perm_r, np.arange(l_matrix.shape[0])] = 1
        Pc[np.arange(l_matrix.shape[0]), lu.perm_c] = 1
        Pr = lil_matrix(Pr)
        Pc = lil_matrix(Pc)
        preconditioner = sla.inv((Pr.T*(L*D*L.T)*Pc.T).tocsc())
        data[idx, 2] = evaluate('ic(0)', l_matrix, rhs, preconditioner)

        # Algebraic MultiGrid preconditioner.
        preconditioner = smoothed_aggregation_solver(
            l_matrix).aspreconditioner(cycle='V')
        preconditioner = csr_matrix(preconditioner.matmat(
            np.eye(shape[0], dtype=np.float32)))
        data[idx, 3] = evaluate('amg', l_matrix, rhs, preconditioner)

        # Learned preconditioner.
        sp_tensor = SparseConvTensor(
            features.T.to(device), coors.int().squeeze(), shape, 1)
        preconditioner = csr_matrix(model(sp_tensor).detach().cpu().numpy())
        data[idx, 4] = evaluate('learned', l_matrix, rhs, preconditioner)

    # np.savetxt('/tmp/time.csv', data[:, :, 0], fmt='%.4f')
    # np.savetxt('/tmp/iterations.csv', data[:, :, 1], fmt='%.4f')
    # np.savetxt('/tmp/condition.csv', data[:, :, 2], fmt='%.4f')

    for m in range(1, 5):
        # Compare time/iterations/condition to baseline CG.
        writer.add_histogram('test/time', data[:, 0, 0]/data[:, m, 0], m)
        writer.add_histogram('test/iterations', data[:, 0, 1]/data[:, m, 1], m)
        writer.add_histogram('test/condition', data[:, 0, 2]/data[:, m, 2], m)
        writer.add_histogram('test/density', data[:, m, 3], m)


def main():
    torch.manual_seed(CFG['SEED'])
    torch.set_num_threads(CFG['N_THREADS'])
    device = torch.device(CFG['DEVICE'])

    model = PrecondNet().to(device)
    model.load_state_dict(torch.load(CFG['LOAD_MODEL']))
    writer = SummaryWriter()

    test(model, writer, device)
    writer.close()
    return True


if __name__ == '__main__':
    main()
