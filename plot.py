import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette="viridis")
viridis = plt.get_cmap("viridis")

# Load data from convergence_2D.npz
convergence_data = np.load('direct_scipy_2D/convergence_2D_direct_scipy.npz')

# Example keys: 'centralised_convergence', 'distributed_convergence', 'centralised_cpu', 'distributed_cpu'
direct_error = convergence_data['e_infty']
cholesky_cpu = convergence_data['cholesky_cpu_times']
forward_backwar_cpu = convergence_data['direct_solver_cpu_times']
n = convergence_data['n']
h = convergence_data['h']

# Discretisation Convergence plot
theoretical_ref = direct_error[-1] * (h / h[-1])**2
plt.figure(figsize=(8, 5))
plt.loglog(n, direct_error, 'o-', label='Discretisation Error', color=viridis(0.2), linewidth=2.2)
plt.loglog(n, theoretical_ref, label='Theoretical O($h^2$)', linestyle='--', color=viridis(0.7), linewidth=2.2)
plt.xlabel('n', fontsize=16)
plt.ylabel('$||u^h-u^h_{ex}||_\infty$', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig('direct_solver_convergence_error_2D.png')
plt.show()

# DirectCPU time plot
plt.figure(figsize=(8, 5))
plt.loglog(n, cholesky_cpu, 'o-',label='Cholesky', color=viridis(0.2), linewidth=2.2)
plt.loglog(n, forward_backwar_cpu, 'o-', label='Forward & Backward', color=viridis(0.7), linewidth=2.2)
theoretical_forward_backward = forward_backwar_cpu[-1] * (n / n[-1])**(3)
theoretical_cholesky = cholesky_cpu[-1] * (n / n[-1])**(4)
plt.loglog(n, theoretical_cholesky, '--', label='Cholesky Theoretical O($n^4$)', color=viridis(0.2), linewidth=2)
plt.loglog(n, theoretical_forward_backward, '--', label='Forward & Backward Theoretical O($n^3$)', color=viridis(0.7), linewidth=2)
plt.xlabel('n', fontsize=16)
plt.ylabel('CPU Time [s]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig('direct_solver_cpu_time_2D.png')
plt.show()


# ### 3D Direct Solver Plots ###
# convergence_data = np.load('convergence_3D_direct.npz')

# direct_error = convergence_data['e_infty']
# cholesky_cpu = convergence_data['cholesky_cpu_times']
# forward_backwar_cpu = convergence_data['direct_solver_cpu_times']
# n = convergence_data['n']
# h = convergence_data['h']

# # Discretisation Convergence plot
# theoretical_ref = direct_error[-1] * (h / h[-1])**2
# plt.figure(figsize=(8, 5))
# plt.loglog(n, direct_error, 'o-', label='Discretisation Error', color=viridis(0.2), linewidth=2.2)
# plt.loglog(n, theoretical_ref, label='Theoretical O($h^2$)', linestyle='--', color=viridis(0.7), linewidth=2.2)
# plt.xlabel('n', fontsize=16)
# plt.ylabel('$||u^h-u^h_{ex}||_\infty$', fontsize=16)
# plt.legend(fontsize=14)
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('direct_solver_convergence_error_3D.png')
# plt.show()

# # DirectCPU time plot
# plt.figure(figsize=(8, 5))
# plt.loglog(n, cholesky_cpu, 'o-',label='Cholesky', color=viridis(0.2), linewidth=2.2)
# plt.loglog(n, forward_backwar_cpu, 'o-', label='Forward & Backward', color=viridis(0.7), linewidth=2.2)
# theoretical_cholesky = cholesky_cpu[-1] * (n / n[-1])**(7)
# theoretical_forward_backward = forward_backwar_cpu[-1] * (n / n[-1])**(5)
# plt.loglog(n, theoretical_cholesky, '--', label='Cholesky Theoretical O($n^7$)', color=viridis(0.2), linewidth=2)
# plt.loglog(n, theoretical_forward_backward, '--', label='Forward & Backward Theoretical O($n^5$)', color=viridis(0.7), linewidth=2)
# plt.xlabel('n', fontsize=16)
# plt.ylabel('CPU Time [s]', fontsize=16)
# plt.legend(fontsize=14)
# plt.grid(True)

# plt.tight_layout()
# plt.savefig('direct_solver_cpu_time_3D.png')
# plt.show()
# ## 3D Direct Solver Plots End ###

# Load data from convergence_2D.npz
convergence_data = np.load('convergence_3D_cg.npz', allow_pickle=True)
iteration_errors = convergence_data['iteration_errors']

# Example keys: 'centralised_convergence', 'distributed_convergence', 'centralised_cpu', 'distributed_cpu'
direct_error = convergence_data['e_infty']
#cholesky_cpu = convergence_data['cholesky_cpu_times']
forward_backwar_cpu = convergence_data['direct_solver_cpu_times']
n = convergence_data['n']
h = convergence_data['h']

# CG IC Convergence plot
plt.figure(figsize=(8, 5))
for i, errors in enumerate(iteration_errors):
    plt.loglog(range(1, len(errors)+1), errors, 'o-', label=f'n={n[i]}', color=viridis(i / len(iteration_errors)), linewidth=2.2)
    plt.xlabel('Iteration m', fontsize=16)
    plt.ylabel('$||r^m||_2 / ||f^h||_2$', fontsize=16)
    plt.grid(True)

plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('cg_solver_convergence_error_2D.png')
plt.show()

# CG IC Iteration number
plt.figure(figsize=(8, 5))
iteration_numbers = [len(errors) for errors in iteration_errors]
theoretical_iteration_numbers = iteration_numbers[0] * (n / n[0])**(1/2)
plt.loglog(n, theoretical_iteration_numbers, '--', label='Theoretical O($n^{1/2}$)', color=viridis(0.5), linewidth=2)
plt.loglog(n, iteration_numbers, 'o-', label='CG Iterations', color=viridis(0.5), linewidth=2.2)
plt.xlabel('n', fontsize=16)
plt.ylabel('Number of Iterations m', fontsize=16)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('cg_solver_iteration_numbers_2D.png')

# CG IC CPU time plot
plt.figure(figsize=(8, 5))
#plt.loglog(n, cholesky_cpu, 'o-',label='Incomplete Cholesky', color=viridis(0.2), linewidth=2.2)
plt.loglog(n, forward_backwar_cpu, 'o-', label='CG', color=viridis(0.7), linewidth=2.2)
# Add theoretical lines
# theoretical_incomplete_cholesky = forward_backwar_cpu[0] * (n / n[0])**(3)
theoretical_cg = forward_backwar_cpu[-1] * (n / n[-1])**(3/2)
# plt.loglog(n, theoretical_incomplete_cholesky, '--', label='Incomplete Cholesky Theoretical O($n^3$)', color=viridis(0.2), linewidth=2)
plt.loglog(n, theoretical_cg, '--', label='CG Theoretical O($n^{3/2}$)', color=viridis(0.7), linewidth=2)
plt.xlabel('n', fontsize=16)
plt.ylabel('CPU Time [s]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig('cg_solver_cpu_time_2D.png')
plt.show()