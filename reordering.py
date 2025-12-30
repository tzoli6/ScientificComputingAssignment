import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

def apply_rcm(A, b):
    """
    Apply Reverse Cuthill–McKee reordering to a symmetric sparse matrix A and RHS b.

    Parameters
    ----------
    A : scipy.sparse matrix (n x n)
        Symmetric sparse system matrix.
    b : array_like, shape (n,)
        Right-hand side vector.

    Returns
    -------
    A_rcm : scipy.sparse.csr_matrix
        Reordered matrix P^T A P.
    b_rcm : numpy.ndarray
        Reordered RHS P^T b (same permutation applied to rows).
    perm : numpy.ndarray
        Permutation array P: indices in the *new* ordering.
    inv_perm : numpy.ndarray
        Inverse permutation: maps reordered solution back to original ordering.
    """
    A_csr = A.tocsr()
    # RCM permutation: gives a permutation such that P^T A P has reduced bandwidth
    perm = reverse_cuthill_mckee(A_csr, symmetric_mode=True)

    # Apply permutation: P^T A P
    A_rcm = A_csr[perm, :][:, perm]

    # Permute RHS: P^T b
    b = np.asarray(b).flatten()
    b_rcm = b[perm]

    # Inverse permutation to map solution back
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))

    return A_rcm, b_rcm, perm, inv_perm
