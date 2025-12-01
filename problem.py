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
        self.A_3D = None
        self.b_1D = None
        self.b_2D = None
        self.b_3D = None

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
    
        h = 1 / (self.n + 1)
        x_ticks = np.arange(1, self.n + 1) * h
        y_ticks = np.arange(1, self.n + 1) * h
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
        if self.A_1D is None:
            print("1D problem not constructed yet. Constructing 1D problem...")
            self.construct_1d_problem()

        self.A_3D = np.kron(np.eye(self.n), np.kron(np.eye(self.n), self.A_1D)) + np.kron(np.kron(np.eye(self.n), self.A_1D), np.eye(self.n)) + np.kron(np.kron(self.A_1D, np.eye(self.n)), np.eye(self.n))

        print("3D Problem Matrix A: \n", self.A_3D.shape)
        print(f"A_3D is symmetric: {np.allclose(self.A_3D.T, self.A_3D)}")
        self.C = self.solver.cholesky_decomposition()
        h = 1 / (self.n + 1)
        x_ticks = np.arange(1, self.n + 1) * h
        y_ticks = np.arange(1, self.n + 1) * h
        z_ticks = np.arange(1, self.n + 1) * h
        
        y_mesh, z_mesh, x_mesh = np.meshgrid(x_ticks, y_ticks, z_ticks)

        x_flattened = x_mesh.flatten()
        y_flattened = y_mesh.flatten()
        z_flattened = z_mesh.flatten()

        # print(f"Flattened x {x_mesh.flatten()}")
        # print(f"Flattened y {y_mesh.flatten()}")
        # print(f"Flattened z {z_mesh.flatten()}")

        self.b_3D = self.f(x_flattened, y_flattened, z_flattened)
        # print("Initial b values w/o bc: \n", self.b_3D)

        h = 1/(self.n+1)
        
        # z=0
        bc_facet_bottom = self.bc(x_flattened[:self.n*self.n], y_flattened[:self.n*self.n].flatten(), 0.)
        self.b_3D[:self.n*self.n] += bc_facet_bottom / (h**2)
        print(f"Z=0 bc correct: {np.all(z_flattened[:self.n*self.n]==0) and len(z_flattened[:self.n*self.n])==self.n**2}")

        # z=1
        bc_facet_top = self.bc(x_flattened[self.n**2*(self.n-1):], y_flattened[self.n**2*(self.n-1):], 1.)
        self.b_3D[self.n**2*(self.n-1):] += bc_facet_top / (h**2)
        print(f"Z=1 bc correct: {np.all(z_flattened[self.n**2*(self.n-1):]==1)}")

        idx_y_0 = np.array([range(k*self.n*self.n, k*self.n*self.n + self.n) for k in range(self.n)]).flatten()
        idx_y_1 = np.array([range(k*self.n*self.n-self.n, k*self.n*self.n) for k in range(self.n)]).flatten()
        idx_y_1 = np.array([range(k*self.n*self.n+self.n*(self.n-1), k*self.n*self.n + self.n*self.n) for k in range(self.n)]).flatten()
        
        # y=0
        bc_bottom = self.bc(x_flattened[idx_y_0], 0., z_flattened[idx_y_0])
        self.b_3D[idx_y_0] += bc_bottom / (h**2)
        print(f"Y=0 bc correct: {np.all(y_flattened[idx_y_0]==0)}")

        # y=1
        bc_top = self.bc(x_flattened[idx_y_1], 1., z_flattened[idx_y_1])
        self.b_3D[idx_y_1] += bc_top / (h**2)
        print(f"Y=1 bc correct: {np.all(y_flattened[idx_y_1]==1)}")

        # x=0
        bc_left = self.bc(0., y_flattened[::self.n], z_flattened[::self.n])
        self.b_3D[::self.n] += bc_left / (h**2)
        print(f"X=0 bc correct: {np.all(x_flattened[::self.n]==0)}")

        # Right boundary (x=1)
        bc_right = self.bc(1, y_flattened[self.n-1::self.n], z_flattened[self.n-1::self.n])
        self.b_3D[self.n-1::self.n] += bc_right / (h**2)
        print(f"X=1 bc correct: {np.all(x_flattened[self.n-1::self.n]==1)}")

        # print("b values: \n", self.b_3D)

    def solve(self):
        # Solve using numpy for verification
        # u = np.linalg.solve(self.A_3D, self.b_3D)
        # print("Solution vector u (numpy): \n", u)

        # Solve using custom solver
        u = self.solver.solve()
        
        if self.A_3D is not None:
            h = 1 / (self.n + 1)
            x_ticks = np.arange(1, self.n + 1) * h
            y_ticks = np.arange(1, self.n + 1) * h
            z_ticks = np.arange(1, self.n + 1) * h
            
            y_mesh, z_mesh, x_mesh = np.meshgrid(x_ticks, y_ticks, z_ticks)

            x_flattened = x_mesh.flatten()
            y_flattened = y_mesh.flatten()
            z_flattened = z_mesh.flatten()
            u_exact = self.bc(x_flattened, y_flattened, z_flattened)
        else:
            h = 1 / (self.n + 1)
            x_ticks = np.arange(1, self.n + 1) * h
            y_ticks = np.arange(1, self.n + 1) * h
            x_mesh, y_mesh = np.meshgrid(x_ticks, y_ticks)

            u_exact = self.bc(x_mesh, y_mesh).flatten()

        e_rms = np.sqrt(np.mean((u - u_exact)**2))
        e_infty = np.max(np.abs(u - u_exact))

        print(f"RMS Error: {e_rms}, Infinity Norm Error: {e_infty}")
        self.u = u

        return u, e_rms, e_infty

    def plot_solution(self):
        if self.A_2D is not None:
            # 2D case
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
            plt.suptitle('Solution to Poisson Problem')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        else:
            # 3D case - plot slices
            h = 1 / (self.n + 1)
            x = np.arange(1, self.n + 1) * h
            y = np.arange(1, self.n + 1) * h
            z = np.arange(1, self.n + 1) * h

            y_mesh, z_mesh, x_mesh = np.meshgrid(x, y, z)

            x_flattened = x_mesh.flatten()
            y_flattened = y_mesh.flatten()
            z_flattened = z_mesh.flatten()

            u_exact = self.bc(x_flattened, y_flattened, z_flattened)
            
            plt.plot(u_exact)
            plt.plot(self.u)
            
            plt.tight_layout()
            plt.show()

def f_example_2D(x, y) -> float:
    return -12*x**2*y**5 - 20*x**4*y**3

def f_example_3D(x, y, z) -> float:
    return -12*x**2*y**5*z**6 - 20*x**4*y**3*z**6 - 30*x**4*y**5*z**4

def bc_2D(x, y) -> float:
    return x**4 * y**5

def bc_3D(x, y, z) -> float:
    return x**4 * y**5 * z**6

# problem = PoissonProblem(15, f_example_3D, bc_3D)
# problem.construct_1d_problem()
# problem.construct_3d_problem()
# problem.solve()
# problem.plot_solution()

# solve for a range of grid sizes, save errors, and plot convergence
ns = list(range(3, 20, 1))
rms_errors = []
inf_errors = []
hs = []
for n in ns:
    print(f"Running n={n}")
    p = PoissonProblem(n, f_example_3D, bc_3D)
    p.construct_1d_problem()
    p.construct_3d_problem()
    try:
        u, e_rms, e_infty = p.solve()
    except Exception as err:
        print(f"solve failed for n={n}: {err}")
        continue
    rms_errors.append(e_rms)
    inf_errors.append(e_infty)
    hs.append(1.0 / (n + 1))

# save results to a numpy file
np.savez("convergence_3D.npz", n=np.array(ns[:len(rms_errors)]), h=np.array(hs),
            e_rms=np.array(rms_errors), e_infty=np.array(inf_errors))

# plot convergence (log-log)
plt.figure()
plt.loglog(hs, rms_errors, 'o-', label='RMS error')
plt.loglog(hs, inf_errors, 's-', label='Infinity error')
# reference O(h^2) line (scaled to first RMS point)
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
plt.savefig('convergence_3D.png', dpi=300, bbox_inches='tight')
plt.show()