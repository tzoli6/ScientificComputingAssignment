import numpy as np
import time
import matplotlib.pyplot as plt
from problem import PoissonProblem, f_example_2D, bc_2D
from conjugate_gradient_solver import ConjugateGradientSolver
from reordering import apply_rcm
import scipy.sparse.linalg as spla

# --------------------------- Test for the ic as BIM -------------------------------
n = 2**5  # example
problem = PoissonProblem(n, f_example_2D, bc_2D)
problem.construct_1d_problem()
problem.construct_2d_problem()   # builds A_2D and b_2D (and Cholesky for direct solve)

A = problem.A_2D
b = problem.b_2D

cg_ic = ConjugateGradientSolver(problem, tol=1e-10, max_iter=5000)
u_ic, cpu_ic, peak_ic = cg_ic.solve_ic_bim(A, b)

# Compare against exact solution and store iteration history
h = 1.0 / (n + 1)
x_ticks = np.arange(1, n + 1) * h
y_ticks = np.arange(1, n + 1) * h
x_mesh, y_mesh = np.meshgrid(x_ticks, y_ticks)
u_exact = bc_2D(x_mesh, y_mesh).flatten()

e_inf = np.max(np.abs(u_ic - u_exact))
print("IC-BIM max error:", e_inf)
print("IC-BIM iterations:", len(cg_ic.error_history))

p = np.arange(2, 6, 1)
ns = np.power(2, p).astype(int)
print(ns)
rms_errors = []
inf_errors = []
direct_solver_cpu_times = []
cholesky_cpu_times = []
peak_memories = []
hs = []

for n in ns:
    print(f"\n-------- ITERATION {n} --------")
    p = PoissonProblem(n, f_example_2D, bc_2D)
    p.construct_1d_problem()
    p.construct_2d_problem()
    u, e_rms, e_infty, cpu_time, peak_memory = p.solve()

    rms_errors.append(e_rms)
    inf_errors.append(e_infty)
    direct_solver_cpu_times.append(cpu_time)
    cholesky_cpu_times.append(p.cholesky_cpu_time)  
    peak_memories.append(peak_memory)
    hs.append(1.0 / (n + 1))
 
# save results to a numpy file
np.savez("convergence_2D.npz", 
            n=np.array(ns[:len(rms_errors)]), h=np.array(hs),
            e_rms=np.array(rms_errors), e_infty=np.array(inf_errors),
            direct_solver_cpu_times=np.array(direct_solver_cpu_times), 
            cholesky_cpu_times=np.array(cholesky_cpu_times),
            peak_memories=np.array(peak_memories))

print(f"rms_errors: {rms_errors}")
print(f"direct_solver_cpu_times: {direct_solver_cpu_times}")

# Plot convergence
plt.figure()
plt.loglog(hs, rms_errors, 'o-', label='RMS error')
plt.loglog(hs, inf_errors, 's-', label='Infinity error')
if len(hs) >= 1:
    h_ref = np.array(hs)
    ref = inf_errors[0] * (h_ref / h_ref[0])**2
    plt.loglog(h_ref, ref, '--', label='O(h^2) reference')
plt.gca().invert_xaxis()
plt.xlabel('h = 1/(n+1)')
plt.ylabel('Error')
plt.title('Convergence of Poisson solver')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.savefig('convergence_2D.png', dpi=300, bbox_inches='tight')

# Plot CPU time vs n
plt.figure()
plt.loglog(ns[:len(direct_solver_cpu_times)], direct_solver_cpu_times, 'o-')
if len(ns) >= 1:
    n_ref = np.array(ns)
    ref = direct_solver_cpu_times[0] * (n_ref / n_ref[0])**(5/2)
    plt.loglog(n_ref, ref, '--', label='O(n^5/2) reference')
plt.xlabel('n (grid size)')
plt.ylabel('CPU time (s)')
plt.title('CPU Time vs Grid Size')
plt.grid(True, which='both', ls='--')
plt.savefig('cpu_time_2D.png', dpi=300, bbox_inches='tight')

# ------------------------- Reordering experiments (Q4–5) -------------------------

# We will compare SciPy's sparse direct solver with and without RCM reordering.
fillin_no_rcm = []
fillin_rcm = []
t_fact_no_rcm = []
t_fact_rcm = []

for n in ns:
    print(f"\n-------- REORDERING EXPERIMENT n = {n} --------")
    p = PoissonProblem(n, f_example_2D, bc_2D)
    p.construct_1d_problem()
    p.construct_2d_problem()

    A = p.A_2D      # assume this is scipy.sparse
    b = p.b_2D

    # Ensure CSC for sparse LU
    A_csc = A.tocsc()

    # --- without reordering ---
    t0 = time.process_time()
    lu = spla.splu(A_csc)
    t_fact = time.process_time() - t0

    nnz_A = A_csc.nnz
    nnz_LU = lu.L.nnz + lu.U.nnz
    fillin_no_rcm.append(nnz_LU / nnz_A)
    t_fact_no_rcm.append(t_fact)

    # --- with RCM reordering ---
    A_rcm, b_rcm, perm, inv_perm = apply_rcm(A, b)
    A_rcm_csc = A_rcm.tocsc()

    t0 = time.process_time()
    lu_rcm = spla.splu(A_rcm_csc)
    t_fact_r = time.process_time() - t0

    nnz_A_rcm = A_rcm_csc.nnz
    nnz_LU_rcm = lu_rcm.L.nnz + lu_rcm.U.nnz
    fillin_rcm.append(nnz_LU_rcm / nnz_A_rcm)
    t_fact_rcm.append(t_fact_r)

    print(f"  no RCM:  factorization time = {t_fact:.4e}s, "
          f"fill-in = {nnz_LU}/{nnz_A} ≈ {fillin_no_rcm[-1]:.2f}")
    print(f"  with RCM: factorization time = {t_fact_r:.4e}s, "
          f"fill-in = {nnz_LU_rcm}/{nnz_A_rcm} ≈ {fillin_rcm[-1]:.2f}")

# Save reordering results
np.savez(
    "reordering_2D.npz",
    n=np.array(ns),
    fillin_no_rcm=np.array(fillin_no_rcm),
    fillin_rcm=np.array(fillin_rcm),
    t_fact_no_rcm=np.array(t_fact_no_rcm),
    t_fact_rcm=np.array(t_fact_rcm),
)
