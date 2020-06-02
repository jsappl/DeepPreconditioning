"""Define different loss functions for training PrecondNet."""
import torch

# TODO: http://www2.cs.cas.cz/semincm/lectures/2010-07-1920-DuintjerTebbens.pdf
# TODO: https://arxiv.org/pdf/1202.1490.pdf
# TODO: https://arxiv.org/pdf/1301.1107v6.pdf


def condition_loss(l_matrix, preconditioner):
    """Compute the analytical condition number with SVD."""
    _, sigma, _ = torch.svd(torch.sparse.mm(l_matrix, preconditioner))
    return sigma[0]/sigma[-1]


def cholesky_iteration_loss(l_matrix, preconditioner, n_iter=8):
    """Cholesky iterations for positive (semi-)definite matrices.

    Krishnamoorthy, Aravindh, and Kenan Kocagoez. "Singular values using
    cholesky decomposition." arXiv preprint arXiv:1202.1490 (2012).
    """
    # Algorithm 2.
    J_k = torch.sparse.mm(l_matrix, preconditioner)
    for _ in range(n_iter):
        R_k = torch.cholesky(J_k, upper=True)
        J_k = torch.mm(R_k, R_k.t())
    sigma = sorted(J_k.diag())
    return sigma[-1]/sigma[0]


def laguerre_loss(l_matrix, preconditioner):
    """Sharp lower bound for smallest eigenvalue to estimate condition number.

    Yamamoto, Y. (2017). On the optimality and sharpness of Laguerre's lower
    bound on the smallest eigenvalue of a symmetric positive definite matrix.
    Applications of Mathematics, 62(4), 319-331.
    https://doi.org/10.21136/AM.2017.0022-17
    """
    preconditioned = torch.sparse.mm(l_matrix, preconditioner)
    m = preconditioned.shape[-1]
    inverse = preconditioned.inverse()
    tr = inverse.trace()

    # Equation (2.10).
    low_bound = m/tr * \
        (1+torch.sqrt((m-1)*(m*torch.mm(inverse, inverse).trace()/tr**2-1)))

    # Gershgorin upper bound.
    diag = preconditioned.diagonal(dim1=-2, dim2=-1)
    radii = torch.sum(preconditioned.abs(), dim=-1) - diag.abs()
    up_bound = torch.max(diag+radii)

    return up_bound/low_bound


def trace_loss(l_matrix, preconditioner):
    """Bound eigenvalues using traces of matrices.

    Wolkowicz, Henry & P.H. Styan, George (1980). Bounds for eigenvalues using
    traces. Linear Algebra and its Applications. 29. 471-506.
    10.1016/0024-3795(80)90258-X.
    """
    preconditioned = torch.sparse.mm(l_matrix, preconditioner)
    n = preconditioned.shape[-1]
    tr = preconditioned.trace()
    tr_of_squared = torch.mm(preconditioned, preconditioned).trace()

    # Corollary 2.1 (ii).
    # m = tr/n
    # s = torch.sqrt(tr_of_squared/n-m**2)
    # if not tr > 0 or not tr**2 > (n-1)*tr_of_squared:
    #     raise ValueError('Conditions of Corollary 2.1 (ii) not fulfilled.')
    # return 1 + (2*s*(n-1)**(1/2)) / (m-s*(n-1)**(1/2))

    p = tr**2/tr_of_squared - (n-1)
    if not tr > 0 or not p > 0:
        raise ValueError('Conditions of Theorem 2.6 not fulfilled.')
    return (1+(1-p**2)*(1/2)) / p
