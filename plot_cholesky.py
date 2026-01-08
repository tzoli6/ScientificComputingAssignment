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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette="viridis")
viridis = plt.get_cmap("viridis")


def plot_cholesky_cpu_2d_with_rcm():
    base = np.load("cholesky_2D.npz")
    rcm  = np.load("cholesky_rcm_2D.npz")

    n = base["n"]
    cpu_fact = base["cpu_fact"]
    cpu_tri  = base["cpu_tri"]
    cpu_tot  = cpu_fact + cpu_tri

    n_rcm = rcm["n"]
    cpu_fact_rcm = rcm["cpu_fact"]
    cpu_tri_rcm  = rcm["cpu_tri"]
    cpu_tot_rcm  = cpu_fact_rcm + cpu_tri_rcm

    plt.figure(figsize=(8, 5))
    plt.loglog(n, cpu_tot, "o-", label="Direct (original)",
               color=viridis(0.2), linewidth=2.2)
    plt.loglog(n_rcm, cpu_tot_rcm, "s--", label="Direct + RCM",
               color=viridis(0.6), linewidth=2.2)

    # Same reference line as in your original plot, e.g. O(n^3)
    ref = cpu_tot[-1] * (n / n[-1])**3
    plt.loglog(n, ref, "--", color="black", label=r"$\mathcal{O}(n^3)$")

    plt.xlabel(r"$n$", fontsize=16)
    plt.ylabel("CPU Time [s]", fontsize=16)
    plt.title("CPU time of banded Cholesky (2D): original vs RCM", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("cholesky_cpu_2D_rcm_compare.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_cholesky_fillin_2d_with_rcm():
    base = np.load("cholesky_2D.npz")
    rcm  = np.load("cholesky_rcm_2D.npz")

    n = base["n"]
    fill  = base["fill_ratio"]
    n_rcm = rcm["n"]
    fill_rcm = rcm["fill_ratio"]

    plt.figure(figsize=(8, 5))
    plt.loglog(n, fill, "o-", label="Direct (original)",
               color=viridis(0.2), linewidth=2.2)
    plt.loglog(n_rcm, fill_rcm, "s--", label="Direct + RCM",
               color=viridis(0.6), linewidth=2.2)

    plt.xlabel(r"$n$", fontsize=16)
    plt.ylabel(r"$\mathrm{nnz}(C)/\mathrm{nnz}(A)$", fontsize=16)
    plt.title("Fill-in ratio of Cholesky factor (2D): original vs RCM", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("cholesky_fillin_2D_rcm_compare.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_cholesky_cpu_3d_with_rcm():
    base = np.load("cholesky_3D.npz")
    rcm  = np.load("cholesky_rcm_3D.npz")

    n = base["n"]
    cpu_fact = base["cpu_fact"]
    cpu_tri  = base["cpu_tri"]
    cpu_tot  = cpu_fact + cpu_tri

    n_rcm = rcm["n"]
    cpu_fact_rcm = rcm["cpu_fact"]
    cpu_tri_rcm  = rcm["cpu_tri"]
    cpu_tot_rcm  = cpu_fact_rcm + cpu_tri_rcm

    plt.figure(figsize=(8, 5))
    plt.loglog(n, cpu_tot, "o-", label="Direct (original)",
               color=viridis(0.2), linewidth=2.2)
    plt.loglog(n_rcm, cpu_tot_rcm, "s--", label="Direct + RCM",
               color=viridis(0.6), linewidth=2.2)

    ref = cpu_tot[-1] * (n / n[-1])**5   # if you used O(n^5) in 3D
    plt.loglog(n, ref, "--", color="black", label=r"$\mathcal{O}(n^5)$")

    plt.xlabel(r"$n$", fontsize=16)
    plt.ylabel("CPU Time [s]", fontsize=16)
    plt.title("CPU time of banded Cholesky (3D): original vs RCM", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("cholesky_cpu_3D_rcm_compare.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_cholesky_cpu_3d_with_rcm()
    plot_cholesky_cpu_2d_with_rcm()