import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette="viridis")
viridis = plt.get_cmap("viridis")

# ---------- 2D IC-BIM residuals ----------
data_2d = np.load("ic_bim_2D.npz", allow_pickle=True)
n_2d = data_2d["n"]
error_histories_2d = data_2d["error_histories"]

plt.figure(figsize=(8, 5))
for i, (n, errors) in enumerate(zip(n_2d, error_histories_2d)):
    # Pick a few representative n to keep the plot readable
    if n not in [16, 64, 128]:
        continue
    iters = range(1, len(errors) + 1)
    plt.loglog(iters, errors, "o-",
               label=fr"$n={n}$",
               color=viridis(i / len(error_histories_2d)),
               linewidth=2.2)

plt.xlabel("Iteration $m$", fontsize=16)
plt.ylabel(r"$\|r^m\|_2 / \|f_h\|_2$", fontsize=16)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("ic_bim_residuals_2D.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- 3D IC-BIM residuals ----------
data_3d = np.load("ic_bim_3D.npz", allow_pickle=True)
n_3d = data_3d["n"]
error_histories_3d = data_3d["error_histories"]

plt.figure(figsize=(8, 5))
for i, (n, errors) in enumerate(zip(n_3d, error_histories_3d)):
    if n not in [8, 32, 64]:
        continue
    iters = range(1, len(errors) + 1)
    plt.loglog(iters, errors, "o-",
               label=fr"$n={n}$",
               color=viridis(i / len(error_histories_3d)),
               linewidth=2.2)

plt.xlabel("Iteration $m$", fontsize=16)
plt.ylabel(r"$\|r^m\|_2 / \|f_h\|_2$", fontsize=16)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("ic_bim_residuals_3D.png", dpi=300, bbox_inches="tight")
plt.show()
