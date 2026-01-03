import numpy as np
import matplotlib.pyplot as plt
from problem import PoissonProblem, f_example_2D, bc_2D, f_example_3D, bc_3D


p = np.arange(2, 12, 1)
ns = np.power(2, p).astype(int)
# ns= np.arange(2, 64, 2)
print(ns)
rms_errors = []
inf_errors = []
direct_solver_cpu_times = []
cholesky_cpu_times = []
peak_memories = []
hs = []
iteration_errors = []

for n in ns:
    print(f"\n-------- ITERATION {n} --------")
    p = PoissonProblem(n, f_example_2D, bc_2D)
    p.construct_1d_problem()
    p.construct_2d_problem()

    u, e_rms, e_infty, cpu_time, peak_memory = p.cg_solve()

    rms_errors.append(e_rms)
    inf_errors.append(e_infty)
    iteration_errors.append(p.solver_cg.error_history)
    direct_solver_cpu_times.append(cpu_time)
    cholesky_cpu_times.append(p.cholesky_cpu_time)  
    peak_memories.append(peak_memory)
    hs.append(1.0 / n)

    # save results to a numpy file after each iteration
    np.savez(f"convergence_3D_cg.npz", 
            n=np.array(ns[:len(rms_errors)]), h=np.array(hs),
            e_rms=np.array(rms_errors), e_infty=np.array(inf_errors),
            iteration_errors=np.array(iteration_errors, dtype=object),
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
# plt.savefig('convergence_2D_direct_scipy.png', dpi=300, bbox_inches='tight')

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
# plt.savefig('cpu_time_2D_direct_scipy.png', dpi=300, bbox_inches='tight')