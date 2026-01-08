import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def estimate_half_bandwidth(A: sp.spmatrix) -> int:
    """
    Estimate the (half) bandwidth of a sparse matrix A.

    For a matrix A = [a_ij], the half bandwidth is
        bw = max_{a_ij != 0} |i - j|.
    """
    A = A.tocsr()
    rows, cols = A.nonzero()
    if rows.size == 0:
        return 0
    return int(np.max(np.abs(rows - cols)))


def apply_rcm_reordering(A: sp.spmatrix, b: np.ndarray):
    """
    Apply Reverse Cuthill–McKee reordering to A to reduce its bandwidth.

    Parameters
    ----------
    A : scipy.sparse matrix
        Symmetric sparse system matrix.
    b : numpy.ndarray
        Right-hand side vector.

    Returns
    -------
    A_rcm : scipy.sparse.csr_matrix
        Reordered matrix P A P^T.
    b_rcm : numpy.ndarray
        Reordered right-hand side P b.
    perm : numpy.ndarray
        Permutation vector P such that A_rcm = A[perm, :][:, perm].
    inv_perm : numpy.ndarray
        Inverse permutation, so that x = x_rcm[inv_perm] recovers the
        solution in the original ordering.
    """
    # Compute RCM permutation (for symmetric matrices)
    perm = reverse_cuthill_mckee(A, symmetric_mode=True)

    # Apply permutation: A_rcm = P A P^T, b_rcm = P b
    A_rcm = A[perm, :][:, perm].tocsr()
    b_rcm = b[perm]

    # Inverse permutation to map solution back
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))

    return A_rcm, b_rcm, perm, inv_perm