# run_cholesky_off_the_shelf.py
import numpy as np
import scipy.sparse as sp

from problem import (
    PoissonProblem,
    f_example_2D,
    f_example_3D,
    bc_2D,
    bc_3D,
)
from solver import Solver


def run_cholesky_2d():
    """
    Generate cholesky_2D.npz using the off-the-shelf dense Cholesky
    methods in Solver (no explicit reordering).
    """
    # Same p-range as before for 2D (adjust if needed)
    p_vals = np.arange(2, 8)      # n = 4, 8, ..., 512
    ns = (2 ** p_vals).astype(int)

    cpu_fact = []
    cpu_tri = []
    fill_ratio = []

    for n in ns:
        print(f"\n===== 2D off-the-shelf Cholesky: n = {n} =====")

        prob = PoissonProblem(n, f_example_2D, bc_2D)
        prob.construct_1d_problem()
        prob.construct_2d_problem()

        solver = Solver(prob)

        # Factorisation (dense off-the-shelf)
        C, t_fact, peak_fact = solver.cholesky_decomposition_off_the_shelf()

        # Solve (two triangular solves)
        u, t_tri, peak_tri = solver.solve_off_the_shelf()

        # Fill-in ratio nnz(C) / nnz(A)
        A = prob.A_2D
        nnz_A = A.nnz if sp.issparse(A) else np.count_nonzero(A)
        nnz_C = np.count_nonzero(C)
        fill = nnz_C / nnz_A

        cpu_fact.append(t_fact)
        cpu_tri.append(t_tri)
        fill_ratio.append(fill)

        print(f"  t_fact = {t_fact:.4e} s, t_tri = {t_tri:.4e} s, "
              f"fill ratio = {fill:.3e}")

    np.savez(
        "cholesky_2D.npz",
        p=p_vals,
        n=ns,
        cpu_fact=np.array(cpu_fact),
        cpu_tri=np.array(cpu_tri),
        fill_ratio=np.array(fill_ratio),
    )
    print("Saved 2D off-the-shelf Cholesky data to cholesky_2D.npz")


def run_cholesky_3d():
    """
    Generate cholesky_3D.npz using the off-the-shelf dense Cholesky
    methods in Solver (no explicit reordering).
    """
    # Keep 3D range modest; dense N^3 explodes quickly
    p_vals = np.arange(2, 6)      # n = 4, 8, 16, 32, 64 (adjust if needed)
    ns = (2 ** p_vals).astype(int)

    cpu_fact = []
    cpu_tri = []
    fill_ratio = []

    for n in ns:
        print(f"\n===== 3D off-the-shelf Cholesky: n = {n} =====")

        prob = PoissonProblem(n, f_example_3D, bc_3D)
        prob.construct_1d_problem()
        prob.construct_3d_problem()

        solver = Solver(prob)

        C, t_fact, peak_fact = solver.cholesky_decomposition_off_the_shelf()
        u, t_tri, peak_tri = solver.solve_off_the_shelf()

        A = prob.A_3D
        nnz_A = A.nnz if sp.issparse(A) else np.count_nonzero(A)
        nnz_C = np.count_nonzero(C)
        fill = nnz_C / nnz_A

        cpu_fact.append(t_fact)
        cpu_tri.append(t_tri)
        fill_ratio.append(fill)

        print(f"  t_fact = {t_fact:.4e} s, t_tri = {t_tri:.4e} s, "
              f"fill ratio = {fill:.3e}")

    np.savez(
        "cholesky_3D.npz",
        p=p_vals,
        n=ns,
        cpu_fact=np.array(cpu_fact),
        cpu_tri=np.array(cpu_tri),
        fill_ratio=np.array(fill_ratio),
    )
    print("Saved 3D off-the-shelf Cholesky data to cholesky_3D.npz")


if __name__ == "__main__":
    run_cholesky_2d()
    run_cholesky_3d()
