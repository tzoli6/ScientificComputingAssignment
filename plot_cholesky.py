import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette="viridis")
viridis = plt.get_cmap("viridis")

# Load data from convergence_2D.npz
sparse_scipy_data = np.load('direct_scipy_2D/convergence_2D_direct_scipy.npz')
sparse_custom_data = np.load('direct_2D/convergence_2D_direct.npz')
dense_custom_data = np.load('direct_dense_2D/convergence_2D_direct_dense.npz')

# Example keys: 'centralised_convergence', 'distributed_convergence', 'centralised_cpu', 'distributed_cpu'
sparse_scipy_cpu_time = sparse_scipy_data['cholesky_cpu_times']
sparse_scipy_n = sparse_scipy_data['n']
sparse_custom_cpu_time = sparse_custom_data['cholesky_cpu_times']
sparse_custom_n = sparse_custom_data['n']
dense_custom_cpu_time = dense_custom_data['cholesky_cpu_times']
dense_custom_n = dense_custom_data['n']

# Discretisation Convergence plot
theoretical_scipy = sparse_scipy_cpu_time[-1] * (sparse_scipy_n / sparse_scipy_n[-1])**(4)
theoretical_custom = sparse_custom_cpu_time[-1] * (sparse_custom_n / sparse_custom_n[-1])**(4)
theoretical_dense = dense_custom_cpu_time[-1] * (dense_custom_n / dense_custom_n[-1])**(4)

plt.figure(figsize=(8, 5))
plt.loglog(sparse_scipy_n, theoretical_scipy, label='Theoretical O($n^4$)', linestyle='--', color=viridis(0.7), linewidth=2.2)
plt.loglog(sparse_custom_n, theoretical_custom, linestyle='--', color=viridis(0.7), linewidth=2.2)
plt.loglog(dense_custom_n, theoretical_dense, linestyle='--', color=viridis(0.7), linewidth=2.2)

plt.loglog(sparse_scipy_n, sparse_scipy_cpu_time, 'o-', label='Off-the-shelf', color=viridis(0.2), linewidth=2.2)
plt.loglog(sparse_custom_n, sparse_custom_cpu_time, 'o-', label='Custom Sparse', color=viridis(0.5), linewidth=2.2)
plt.loglog(dense_custom_n, dense_custom_cpu_time, 'o-', label='Custom Dense', color=viridis(0.9), linewidth=2.2)

plt.xlabel('n', fontsize=16)
plt.ylabel('CPU Time [s]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig('cholesky_2D.png')
plt.show()

# Load data from 3D data
sparse_scipy_data = np.load('direct_scipy_3D/convergence_3D_direct_scipy.npz')
sparse_custom_data = np.load('direct_3D/convergence_3D_direct.npz')
dense_custom_data = np.load('direct_dense_3D/convergence_3D_direct_dense.npz')

# Example keys: 'centralised_convergence', 'distributed_convergence', 'centralised_cpu', 'distributed_cpu'
sparse_scipy_cpu_time = sparse_scipy_data['cholesky_cpu_times']
sparse_scipy_n = sparse_scipy_data['n']
sparse_custom_cpu_time = sparse_custom_data['cholesky_cpu_times']
sparse_custom_n = sparse_custom_data['n']
dense_custom_cpu_time = dense_custom_data['cholesky_cpu_times']
dense_custom_n = dense_custom_data['n']

# Discretisation Convergence plot
theoretical_scipy = sparse_scipy_cpu_time[-1] * (sparse_scipy_n / sparse_scipy_n[-1])**(7)
theoretical_custom = sparse_custom_cpu_time[-1] * (sparse_custom_n / sparse_custom_n[-1])**(7)
theoretical_dense = dense_custom_cpu_time[-1] * (dense_custom_n / dense_custom_n[-1])**(7)

plt.figure(figsize=(8, 5))
plt.loglog(sparse_scipy_n, theoretical_scipy, label='Theoretical O($n^7$)', linestyle='--', color=viridis(0.7), linewidth=2.2)
plt.loglog(sparse_custom_n, theoretical_custom, linestyle='--', color=viridis(0.7), linewidth=2.2)
plt.loglog(dense_custom_n, theoretical_dense, linestyle='--', color=viridis(0.7), linewidth=2.2)

plt.loglog(sparse_scipy_n, sparse_scipy_cpu_time, 'o-', label='Off-the-shelf', color=viridis(0.2), linewidth=2.2)
plt.loglog(sparse_custom_n, sparse_custom_cpu_time, 'o-', label='Custom Sparse', color=viridis(0.5), linewidth=2.2)
plt.loglog(dense_custom_n, dense_custom_cpu_time, 'o-', label='Custom Dense', color=viridis(0.9), linewidth=2.2)

plt.xlabel('n', fontsize=16)
plt.ylabel('CPU Time [s]', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.savefig('cholesky_3D.png')
plt.show()