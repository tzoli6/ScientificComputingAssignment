import numpy as np
import scipy.sparse as sp

from problem import (
    PoissonProblem,
    f_example_2D,
    f_example_3D,
    bc_2D,
    bc_3D,
)
from reordering import apply_rcm_reordering, estimate_half_bandwidth
from solver import Solver


def run_cholesky_rcm_2d():
    """
    Run banded Cholesky (sparse_fast + solve_sparse) on RCM-reordered
    2D systems, and save bandwidth, CPU times and fill-in.
    """
    # Use the same p-range as for the original direct-solver study
    p_vals = np.arange(2, 8, 1)
    ns = (2 ** p_vals).astype(int)

    bw_before = []
    bw_after = []
    cpu_fact_rcm = []
    cpu_tri_rcm = []
    fill_ratio_rcm = []

    for n in ns:
        print(f"\n===== 2D Cholesky + RCM: n = {n} =====")
        prob = PoissonProblem(n, f_example_2D, bc_2D)
        prob.construct_1d_problem()
        prob.construct_2d_problem()

        A = prob.A_2D.tocsr()
        b = prob.b_2D

        # Bandwidth before reordering
        bw_before.append(estimate_half_bandwidth(A))

        # Apply RCM reordering
        A_rcm, b_rcm, perm, inv_perm = apply_rcm_reordering(A, b)
        bw_after.append(estimate_half_bandwidth(A_rcm))

        # Overwrite the problem matrix and RHS with the reordered ones
        prob.A_2D = A_rcm
        prob.b_2D = b_rcm

        # Use your existing Solver on the reordered system
        solver = Solver(prob)

        # Factorisation (sparse banded)
        C, t_fact, peak_fact = solver.cholesky_decomposition_off_the_shelf()

        # Triangular solves (banded)
        u_rcm, t_tri, peak_tri = solver.solve_off_the_shelf()

        # Fill-in ratio nnz(C) / nnz(A)
        nnz_A = A_rcm.nnz
        nnz_C = solver.C.nnz if sp.issparse(solver.C) else np.count_nonzero(solver.C)
        fill_ratio_rcm.append(nnz_C / nnz_A)

        cpu_fact_rcm.append(t_fact)
        cpu_tri_rcm.append(t_tri)

        # Map solution back to original ordering if you want to check errors
        u = u_rcm[inv_perm]
        # (Optionally compare u with exact solution here.)

        print(f"  bandwidth before: {bw_before[-1]}, after: {bw_after[-1]}")
        print(f"  t_fact = {t_fact:.4e} s, t_tri = {t_tri:.4e} s, "
              f"fill ratio = {fill_ratio_rcm[-1]:.3f}")

    np.savez(
        "cholesky_rcm_2D.npz",
        p=p_vals,
        n=ns,
        bw_before=np.array(bw_before),
        bw_after=np.array(bw_after),
        cpu_fact=np.array(cpu_fact_rcm),
        cpu_tri=np.array(cpu_tri_rcm),
        fill_ratio=np.array(fill_ratio_rcm),
    )
    print("Saved reordered 2D Cholesky data to cholesky_rcm_2D.npz")


def run_cholesky_rcm_3d():
    """
    Same as above, but for the 3D Poisson problem.
    """
    p_vals = np.arange(2, 6, 1)  # same as your original 3D study
    ns = (2 ** p_vals).astype(int)

    bw_before = []
    bw_after = []
    cpu_fact_rcm = []
    cpu_tri_rcm = []
    fill_ratio_rcm = []

    for n in ns:
        print(f"\n===== 3D Cholesky + RCM: n = {n} =====")
        prob = PoissonProblem(n, f_example_3D, bc_3D)
        prob.construct_1d_problem()
        prob.construct_3d_problem()

        A = prob.A_3D.tocsr()
        b = prob.b_3D

        bw_before.append(estimate_half_bandwidth(A))
        A_rcm, b_rcm, perm, inv_perm = apply_rcm_reordering(A, b)
        bw_after.append(estimate_half_bandwidth(A_rcm))

        prob.A_3D = A_rcm
        prob.b_3D = b_rcm

        solver = Solver(prob)
        C, t_fact, peak_fact = solver.cholesky_decomposition_off_the_shelf()
        u_rcm, t_tri, peak_tri = solver.solve_off_the_shelf()

        nnz_A = A_rcm.nnz
        nnz_C = solver.C.nnz if sp.issparse(solver.C) else np.count_nonzero(solver.C)
        fill_ratio_rcm.append(nnz_C / nnz_A)

        cpu_fact_rcm.append(t_fact)
        cpu_tri_rcm.append(t_tri)

        u = u_rcm[inv_perm]

        print(f"  bandwidth before: {bw_before[-1]}, after: {bw_after[-1]}")
        print(f"  t_fact = {t_fact:.4e} s, t_tri = {t_tri:.4e} s, "
              f"fill ratio = {fill_ratio_rcm[-1]:.3f}")

    np.savez(
        "cholesky_rcm_3D.npz",
        p=p_vals,
        n=ns,
        bw_before=np.array(bw_before),
        bw_after=np.array(bw_after),
        cpu_fact=np.array(cpu_fact_rcm),
        cpu_tri=np.array(cpu_tri_rcm),
        fill_ratio=np.array(fill_ratio_rcm),
    )
    print("Saved reordered 3D Cholesky data to cholesky_rcm_3D.npz")


if __name__ == "__main__":
    run_cholesky_rcm_2d()
    run_cholesky_rcm_3d()
