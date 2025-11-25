import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from solver import Solver


class PoissonProblem:

    def __init__(self, n, f:callable, bc:callable):
        # self.domain = domain
        # self.boundary_conditions = boundary_conditions
        # self.solution = None
        self.n = n

        self.A_1D = None
        self.A_2D = None
        self.b_1D = None
        self.b_2D = None

        self.solver = Solver(self)
        self.f = f
        self.bc = bc

    def construct_1d_problem(self):
        A_i = -np.array([1, -2, 1])

        # Interior
        self.A_1D = np.diag(A_i[0]*np.ones(self.n-1), k=-1) + \
        np.diag(A_i[1]*np.ones(self.n), k=0) + \
        np.diag(A_i[2]*np.ones(self.n-1), k=1)
        self.A_1D *= (self.n+1)**2
        print("1D Problem Matrix A wo bc: \n", self.A_1D)

        # # Boundary conditions
        # A = np.zeros((self.n+2, self.n+2))
        # A[0, 0] = 1  # Dirichlet BC at x=0
        # A[1, 0] = -1
        # A[-1, -1] = 1  # Dirichlet BC at x=1
        # A[-2, -1] = -1  # Dirichlet BC at x=1
        # A[1:-1, 1:-1] = self.A_1D
        # self.A_1D = A
        # print("1D Problem Matrix A: \n", A)


    def construct_2d_problem(self):
        if self.A_1D is None:
            print("1D problem not constructed yet. Constructing 1D problem...")
            self.construct_1d_problem()

        self.A_2D = np.kron(np.eye(self.n), self.A_1D) + np.kron(self.A_1D, np.eye(self.n))

        print("2D Problem Matrix A: \n", self.A_2D)

        print(f"A_2D is symmetric: {np.allclose(self.A_2D.T, self.A_2D)}")
        # print(f"A_2D is positive definite: {np.all(np.linalg.eigvals(self.A_2D) > 0)}")
        # print(np.linalg.eigvals(self.A_2D))
        self.C = self.solver.cholesky_decomposition()
    

        x_ticks = np.linspace(0, 1, self.n)
        y_ticks = np.linspace(0, 1, self.n)
        x_mesh, y_mesh = np.meshgrid(x_ticks, y_ticks)

        self.b_2D = self.f(x_mesh, y_mesh).flatten()
        print("Initial b values w/o bc: \n", self.b_2D)

        h = 1/(self.n+1)

        # Bottom boundary (y=0)
        bc_bottom = self.bc(x_ticks, 0)
        self.b_2D[0:self.n] += bc_bottom / (h**2)

        # Top boundary (y=1)
        bc_top = self.bc(x_ticks, 1)
        self.b_2D[-self.n:] += bc_top / (h**2)

        # Left boundary (x=0)
        bc_left = self.bc(0, y_ticks)
        self.b_2D[::self.n] += bc_left / (h**2)

        # Right boundary (x=1)
        bc_right = self.bc(1, y_ticks)
        self.b_2D[self.n-1::self.n] += bc_right / (h**2)
        print("b values: \n", self.b_2D)

    def construct_3d_problem(self):
        pass

    def solve(self):
        # Solve using numpy for verification
        u = np.linalg.solve(self.A_2D, self.b_2D)
        print("Solution vector u (numpy): \n", u)

        # Solve using custom solver
        self.u = self.solver.solve()

        x = np.linspace(0, 1, self.n)
        y = np.linspace(0, 1, self.n)
        X, Y = np.meshgrid(x, y)

        u_exact = self.bc(X, Y).flatten()

        e_rms = np.sqrt(np.mean((u - u_exact)**2))
        e_infty = np.max(np.abs(u - u_exact))

        return self.u, e_rms, e_infty

    def plot_solution(self):
        u_reshaped = self.u.reshape((self.n, self.n))
        x = np.linspace(0, 1, self.n)
        y = np.linspace(0, 1, self.n)
        X, Y = np.meshgrid(x, y)

        u_exact = self.bc(X, Y)

        plt.figure(figsize=(10, 8))
        plt.subplot(1, 3, 1)
        sns.heatmap(u_reshaped, xticklabels=np.round(x, 2), yticklabels=np.round(y, 2), cmap='viridis', cbar_kws={'label': 'u'})
        plt.title('Simulated Solution')
        
        plt.subplot(1, 3, 2)
        sns.heatmap(u_exact, xticklabels=np.round(x, 2), yticklabels=np.round(y, 2), cmap='viridis', cbar_kws={'label': 'u'})
        plt.title('Exact Solution')
        
        plt.subplot(1, 3, 3)
        sns.heatmap(u_reshaped - u_exact, xticklabels=np.round(x, 2), yticklabels=np.round(y, 2), cmap='viridis', cbar_kws={'label': 'error'})
        plt.title('Error')
        plt.title('Solution to Poisson Problem')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

def f_example(x, y) -> float:
    return -12*x**2*y**5 - 20*x**4*y**3

def bc_example(x, y) -> float:
    return x**4 * y**5
# for n in range(1, 41, 10):
#     problem = PoissonProblem(50, f_example, bc_example)
#     problem.construct_1d_problem()
#     problem.construct_2d_problem()
#     # solve for a range of grid sizes, save errors, and plot convergence
ns = list(range(10, 1000, 10))
rms_errors = []
inf_errors = []
hs = []
for n in ns:
    print(f"Running n={n}")
    p = PoissonProblem(n, f_example, bc_example)
    p.construct_1d_problem()
    p.construct_2d_problem()
    try:
        u, e_rms, e_infty = p.solve()
    except Exception as err:
        print(f"solve failed for n={n}: {err}")
        continue
    rms_errors.append(e_rms)
    inf_errors.append(e_infty)
    hs.append(1.0 / (n + 1))

# save results to a numpy file
np.savez("convergence.npz", n=np.array(ns[:len(rms_errors)]), h=np.array(hs),
            e_rms=np.array(rms_errors), e_infty=np.array(inf_errors))

# plot convergence (log-log)
plt.figure()
plt.loglog(hs, rms_errors, 'o-', label='RMS error')
plt.loglog(hs, inf_errors, 's-', label='Infinity error')
# reference O(h^2) line (scaled to first RMS point)
if len(hs) >= 1:
    h_ref = np.array(hs)
    ref = rms_errors[0] * (h_ref / h_ref[0])**2
    plt.loglog(h_ref, ref, '--', label='O(h^2) reference')
plt.gca().invert_xaxis()
plt.xlabel('h = 1/(n+1)')
plt.ylabel('Error')
plt.title('Convergence of Poisson solver')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()