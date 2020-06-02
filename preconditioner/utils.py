"""Utility functions needed througout the training/testing phase."""
from time import time

import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse import coo_matrix


def _time_cg(l_matrix, rhs, preconditioner=None):
    n_iter = 0
    maxiter = 1024
    residuals = np.empty((maxiter, 2))

    def callback(xk):
        nonlocal n_iter
        residuals[n_iter] = [n_iter+1, np.sum((l_matrix*xk-rhs)**2)]
        n_iter += 1

    if preconditioner is None:
        t0 = time()
        _, _ = sla.cg(l_matrix, rhs, maxiter=maxiter, callback=callback)
        t1 = time()
    else:
        t0 = time()
        _, _ = sla.cg(l_matrix, rhs, M=preconditioner,
                      maxiter=maxiter, callback=callback)
        t1 = time()

    return residuals, n_iter, t1-t0


def evaluate(method, l_matrix, rhs, preconditioner=None):
    res, n_iter, time = _time_cg(l_matrix, rhs, preconditioner)
    # np.savetxt('./residual_'+method+'.csv',
    #            res[:n_iter], fmt='%.32f', header='it,res', delimiter=',')

    sigma = np.linalg.svd(l_matrix.dot(
        preconditioner).toarray(), compute_uv=False)
    cond_number = sigma[0]/sigma[-1]
    density = preconditioner.nnz/np.prod(preconditioner.shape)*100

    return time, n_iter, cond_number, density


def is_positive_definite(l_matrix_csv):
    data = np.genfromtxt(l_matrix_csv, delimiter=',')
    row = data[:, 0]
    col = data[:, 1]
    val = -data[:, 2]  # make if SPD for CG method
    n_rows = int(max(row))+1
    l_matrix = coo_matrix((val, (row, col)), shape=(n_rows, n_rows))

    assert ((l_matrix.transpose() != l_matrix).nnz ==
            0), 'Non-symmetric matrix generated!'

    vals, _ = sla.eigs(l_matrix)
    assert ((vals > 0).all()), 'Non-positive definite matrix generated!'

    return l_matrix
