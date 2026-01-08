# plot_cholesky_rcm.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette="viridis")
viridis = plt.get_cmap("viridis")


def plot_bandwidth_2d():
    base = np.load("cholesky_2D.npz")
    rcm = np.load("cholesky_rcm_2D.npz")

    n_base = base["n"]
    n_rcm = rcm["n"]

    # If you stored bw in the original run; otherwise approximate or skip
    bw_base = base.get("bw", None)
    if bw_base is None:
        # Rough proxy: half-bandwidth ~ n for unreordered 2D
        bw_base = n_base

    bw_before = rcm["bw_before"]
    bw_after = rcm["bw_after"]

    plt.figure(figsize=(7, 5))
    # Original: full n-range
    plt.loglog(n_base, bw_base, "o-", label="Original",
               color=viridis(0.2), linewidth=2.0)
    # RCM: only where you actually ran it
    plt.loglog(n_rcm, bw_before, "s--", label="RCM (before)",
               color=viridis(0.5), linewidth=2.0)
    plt.loglog(n_rcm, bw_after, "d-", label="RCM (after)",
               color=viridis(0.8), linewidth=2.0)

    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel("Half bandwidth", fontsize=14)
    plt.title("Half bandwidth vs $n$ (2D)", fontsize=15)
    plt.grid(True, which="both")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("cholesky_bandwidth_2D_rcm_compare.png",
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_fillin_2d():
    base = np.load("cholesky_2D.npz")
    rcm  = np.load("cholesky_rcm_2D.npz")

    n = base["n"]
    fill_base = base["fill_ratio"]
    fill_rcm  = rcm["fill_ratio"]

    plt.figure(figsize=(7, 5))
    plt.loglog(n, fill_base, "o-", label="Original",
               color=viridis(0.3), linewidth=2.0)
    plt.loglog(n, fill_rcm, "s--", label="RCM",
               color=viridis(0.7), linewidth=2.0)

    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"$\mathrm{nnz}(C)/\mathrm{nnz}(A)$", fontsize=14)
    plt.title("Fill-in ratio of Cholesky factor (2D)", fontsize=15)
    plt.grid(True, which="both")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("cholesky_fillin_2D_rcm_compare.png",
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_cpu_2d():
    base = np.load("cholesky_2D.npz")
    rcm  = np.load("cholesky_rcm_2D.npz")

    n = base["n"]
    cpu_fact_base = base["cpu_fact"]
    cpu_tri_base  = base["cpu_tri"]
    cpu_tot_base  = cpu_fact_base + cpu_tri_base

    n_rcm = rcm["n"]
    cpu_fact_rcm = rcm["cpu_fact"]
    cpu_tri_rcm  = rcm["cpu_tri"]
    cpu_tot_rcm  = cpu_fact_rcm + cpu_tri_rcm

    plt.figure(figsize=(7, 5))
    plt.loglog(n, cpu_tot_base, "o-", label="Original (total)",
               color=viridis(0.2), linewidth=2.0)
    plt.loglog(n_rcm, cpu_tot_rcm, "s--", label="RCM (total)",
               color=viridis(0.7), linewidth=2.0)

    # Same reference line as in your original direct-solver plot, e.g. O(n^3)
    ref = cpu_tot_base[-1] * (n / n[-1])**3
    plt.loglog(n, ref, "--", color="black", label=r"$\mathcal{O}(n^3)$")

    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel("CPU Time [s]", fontsize=14)
    plt.title("CPU time of banded Cholesky (2D): original vs RCM",
              fontsize=15)
    plt.grid(True, which="both")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("cholesky_cpu_2D_rcm_compare.png",
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_bandwidth_3d():
    base = np.load("cholesky_3D.npz")
    rcm  = np.load("cholesky_rcm_3D.npz")
    n_base = base["n"]
    n_rcm = rcm["n"]

    bw_base = base.get("bw", None)
    if bw_base is None:
        # Rough proxy for original 3D ordering
        bw_base = n_base ** 2

    bw_before = rcm["bw_before"]
    bw_after = rcm["bw_after"]

    plt.figure(figsize=(7, 5))
    plt.loglog(n_base, bw_base, "o-", label="Original",
               color=viridis(0.2), linewidth=2.0)
    plt.loglog(n_rcm, bw_before, "s--", label="RCM (before)",
               color=viridis(0.5), linewidth=2.0)
    plt.loglog(n_rcm, bw_after, "d-", label="RCM (after)",
               color=viridis(0.8), linewidth=2.0)

    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel("Half bandwidth", fontsize=14)
    plt.title("Half bandwidth vs $n$ (3D)", fontsize=15)
    plt.grid(True, which="both")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("cholesky_bandwidth_3D_rcm_compare.png",
                dpi=300, bbox_inches="tight")
    plt.show()



def plot_fillin_3d():
    base = np.load("cholesky_3D.npz")
    rcm  = np.load("cholesky_rcm_3D.npz")

    n = base["n"]
    fill_base = base["fill_ratio"]
    fill_rcm  = rcm["fill_ratio"]

    plt.figure(figsize=(7, 5))
    plt.loglog(n, fill_base, "o-", label="Original",
               color=viridis(0.3), linewidth=2.0)
    plt.loglog(n, fill_rcm, "s--", label="RCM",
               color=viridis(0.7), linewidth=2.0)

    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"$\mathrm{nnz}(C)/\mathrm{nnz}(A)$", fontsize=14)
    plt.title("Fill-in ratio of Cholesky factor (3D)", fontsize=15)
    plt.grid(True, which="both")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("cholesky_fillin_3D_rcm_compare.png",
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_cpu_3d():
    base = np.load("cholesky_3D.npz")
    rcm  = np.load("cholesky_rcm_3D.npz")

    n = base["n"]
    cpu_fact_base = base["cpu_fact"]
    cpu_tri_base  = base["cpu_tri"]
    cpu_tot_base  = cpu_fact_base + cpu_tri_base

    n_rcm = rcm["n"]
    cpu_fact_rcm = rcm["cpu_fact"]
    cpu_tri_rcm  = rcm["cpu_tri"]
    cpu_tot_rcm  = cpu_fact_rcm + cpu_tri_rcm

    plt.figure(figsize=(7, 5))
    plt.loglog(n, cpu_tot_base, "o-", label="Original (total)",
               color=viridis(0.2), linewidth=2.0)
    plt.loglog(n_rcm, cpu_tot_rcm, "s--", label="RCM (total)",
               color=viridis(0.7), linewidth=2.0)

    # Same reference line as in your original 3D direct-solver plot,
    # e.g. O(n^5) if that is what you used.
    ref = cpu_tot_base[-1] * (n / n[-1])**5
    plt.loglog(n, ref, "--", color="black", label=r"$\mathcal{O}(n^5)$")

    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel("CPU Time [s]", fontsize=14)
    plt.title("CPU time of banded Cholesky (3D): original vs RCM",
              fontsize=15)
    plt.grid(True, which="both")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("cholesky_cpu_3D_rcm_compare.png",
                dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # 2D
    plot_bandwidth_2d()
    plot_fillin_2d()
    plot_cpu_2d()

    # 3D
    plot_bandwidth_3d()
    plot_fillin_3d()
    plot_cpu_3d()
