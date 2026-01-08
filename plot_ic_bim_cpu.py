import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette="viridis")
viridis = plt.get_cmap("viridis")


def plot_ic_bim_cpu_2d():
    data = np.load("ic_bim_2D.npz", allow_pickle=True)
    n = data["n"]
    cpu_fact = data["cpu_fact"]
    cpu_iter = data["cpu_iter"]
    niters = data["niters"]

    # Ignore runs that did not converge within max_iter
    mask = niters < 10_000
    n = n[mask]
    cpu_fact = cpu_fact[mask]
    cpu_iter = cpu_iter[mask]

    cpu_total = cpu_fact + cpu_iter

    plt.figure(figsize=(7, 4.5))
    plt.loglog(n, cpu_total, "o-", label="IC-BIM total",
               color=viridis(0.3), linewidth=2.2)
    plt.loglog(n, cpu_fact, "s--", label="IC factorisation",
               color=viridis(0.6), linewidth=1.8)
    plt.loglog(n, cpu_iter, "d--", label="IC iterations",
               color=viridis(0.9), linewidth=1.8)

    # Theoretical O(n^4) reference line (2D: cost ~ N^2, N ~ n^2)
    ref = cpu_total[-1] * (n / n[-1])**4
    plt.loglog(n, ref, "--", label=r"Theoretical $\mathcal{O}(n^4)$",
               color="black", linewidth=2.0)

    plt.xlabel(r"$n$", fontsize=16)
    plt.ylabel("CPU Time [s]", fontsize=16)
    plt.title("CPU time of IC-BIM (2D)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("ic_bim_cpu_2D.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_ic_bim_cpu_3d():
    data = np.load("ic_bim_3D.npz", allow_pickle=True)
    n = data["n"]
    cpu_fact = data["cpu_fact"]
    cpu_iter = data["cpu_iter"]
    niters = data["niters"]

    # Ignore runs that did not converge within max_iter, if any
    mask = niters < 10_000
    n = n[mask]
    cpu_fact = cpu_fact[mask]
    cpu_iter = cpu_iter[mask]

    cpu_total = cpu_fact + cpu_iter

    plt.figure(figsize=(7, 4.5))
    plt.loglog(n, cpu_total, "o-", label="IC-BIM total",
               color=viridis(0.3), linewidth=2.2)
    plt.loglog(n, cpu_fact, "s--", label="IC factorisation",
               color=viridis(0.6), linewidth=1.8)
    plt.loglog(n, cpu_iter, "d--", label="IC iterations",
               color=viridis(0.9), linewidth=1.8)

    # Theoretical O(n^5) reference line (3D: cost ~ N^{5/3}, N ~ n^3)
    ref = cpu_total[-1] * (n / n[-1])**5
    plt.loglog(n, ref, "--", label=r"Theoretical $\mathcal{O}(n^5)$",
               color="black", linewidth=2.0)

    plt.xlabel(r"$n$", fontsize=16)
    plt.ylabel("CPU Time [s]", fontsize=16)
    plt.title("CPU time of IC-BIM (3D)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("ic_bim_cpu_3D.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_ic_bim_cpu_2d()
    plot_ic_bim_cpu_3d()
