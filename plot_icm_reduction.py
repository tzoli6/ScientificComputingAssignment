import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette="viridis")
viridis = plt.get_cmap("viridis")


def reduction_factors_from_history(errors, n_last=5):
    """Compute average residual reduction factor over the last n_last steps."""
    errors = np.asarray(errors, dtype=float)
    if len(errors) <= n_last:
        return np.nan
    # Ratios r^m / r^{m-1} for last n_last iterations
    last_idxs = np.arange(len(errors) - n_last + 1, len(errors))
    ratios = errors[last_idxs] / errors[last_idxs - 1]
    return float(np.mean(ratios))


# --------------------- 2D ---------------------
data_2d = np.load("ic_bim_2D.npz", allow_pickle=True)
n_2d = data_2d["n"]
h_2d = data_2d["h"]
error_histories_2d = data_2d["error_histories"]
niters_2d = data_2d["niters"]

red_2d = []
h2_2d = []

for n, h, errors, nit in zip(n_2d, h_2d, error_histories_2d, niters_2d):
    # Optionally skip runs that did not converge within max_iter
    # (here: nit == max_iter)
    if nit >= 10_000:
        continue
    q = reduction_factors_from_history(errors, n_last=5)
    red_2d.append(q)
    h2_2d.append(h ** 2)

red_2d = np.array(red_2d)
h2_2d = np.array(h2_2d)

plt.figure(figsize=(7, 4.5))
plt.plot(h2_2d, 1.0 - red_2d, "o-", linewidth=2.0)
plt.xlabel(r"$h^2$", fontsize=14)
plt.ylabel(r"$1 - q_{\mathrm{IC}}$", fontsize=14)
plt.title("Asymptotic reduction factor of IC-BIM (2D)", fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig("ic_bim_reduction_2D.png", dpi=300, bbox_inches="tight")
plt.show()


# --------------------- 3D ---------------------
data_3d = np.load("ic_bim_3D.npz", allow_pickle=True)
n_3d = data_3d["n"]
h_3d = data_3d["h"]
error_histories_3d = data_3d["error_histories"]
niters_3d = data_3d["niters"]

red_3d = []
h2_3d = []

for n, h, errors, nit in zip(n_3d, h_3d, error_histories_3d, niters_3d):
    if nit >= 10_000:
        continue
    q = reduction_factors_from_history(errors, n_last=5)
    red_3d.append(q)
    h2_3d.append(h ** 2)

red_3d = np.array(red_3d)
h2_3d = np.array(h2_3d)

plt.figure(figsize=(7, 4.5))
plt.plot(h2_3d, 1.0 - red_3d, "o-", linewidth=2.0)
plt.xlabel(r"$h^2$", fontsize=14)
plt.ylabel(r"$1 - q_{\mathrm{IC}}$", fontsize=14)
plt.title("Asymptotic reduction factor of IC-BIM (3D)", fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig("ic_bim_reduction_3D.png", dpi=300, bbox_inches="tight")
plt.show()
