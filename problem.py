import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from solver import Solver
from conjugate_gradient_solver import ConjugateGradientSolver
from scipy import sparse as sp


class PoissonProblem:

    def __init__(self, n, f:callable, bc:callable):
        # self.domain = domain
        # self.boundary_conditions = boundary_conditions
        # self.solution = None
        self.n = n
        self.h = 1 / (n + 1)

        self.A_1D = None
        self.A_2D = None
        self.A_3D = None
        self.b_1D = None
        self.b_2D = None
        self.b_3D = None

        self.cholesky_cpu_time = None
        self.cholesky_peak_memorie = None

        self.solver = Solver(self)
        # self.solver = ConjugateGradientSolver(self)
        self.f = f
        self.bc = bc

    def construct_1d_problem(self):
        A_i = -np.array([1, -2, 1])

        # Interior
        self.A_1D = sp.diags(A_i[0]*np.ones(self.n-1), offsets=-1) + \
        sp.diags(A_i[1]*np.ones(self.n), offsets=0) + \
        sp.diags(A_i[2]*np.ones(self.n-1), offsets=1)
        self.A_1D *= (self.n+1)**2

    def construct_2d_problem(self):
        if self.A_1D is None:
            print("1D problem not constructed yet. Constructing 1D problem...")
            self.construct_1d_problem()

        self.A_2D = sp.kron(sp.eye(self.n), self.A_1D) + sp.kron(self.A_1D, sp.eye(self.n))

        # Cholesky decomposition
        self.C, cpu_time, peak = self.solver.cholesky_decomposition_banded()
        self.cholesky_cpu_time = cpu_time
        self.cholesky_peak_memorie = peak

        # Construct righ-hand side vecor
        h = 1 / (self.n + 1)
        x_ticks = np.arange(1, self.n + 1) * h
        y_ticks = np.arange(1, self.n + 1) * h
        x_mesh, y_mesh = np.meshgrid(x_ticks, y_ticks)

        self.b_2D = self.f(x_mesh, y_mesh).flatten()

        # Bottom boundary (y=0)
        bc_bottom = self.bc(x_ticks, 0)
        self.b_2D[0:self.n] += bc_bottom / (self.h**2)
        # Top boundary (y=1)
        bc_top = self.bc(x_ticks, 1)
        self.b_2D[-self.n:] += bc_top / (self.h**2)
        # Left boundary (x=0)
        bc_left = self.bc(0, y_ticks)
        self.b_2D[::self.n] += bc_left / (self.h**2)
        # Right boundary (x=1)
        bc_right = self.bc(1, y_ticks)
        self.b_2D[self.n-1::self.n] += bc_right / (self.h**2)

    def construct_3d_problem(self):
        if self.A_1D is None:
            print("1D problem not constructed yet. Constructing 1D problem...")
            self.construct_1d_problem()

        self.A_3D = sp.kron(sp.eye(self.n), sp.kron(sp.eye(self.n), self.A_1D)) + sp.kron(sp.kron(sp.eye(self.n), self.A_1D), sp.eye(self.n)) + sp.kron(sp.kron(self.A_1D, sp.eye(self.n)), sp.eye(self.n))

        # Cholesky decomposition
        self.C, cpu_time, peak = self.solver.cholesky_decomposition_banded()
        self.cholesky_cpu_time = cpu_time
        self.cholesky_peak_memorie = peak

        # Construct righ-hand side vecor  
        x_ticks = np.arange(1, self.n + 1) * self.h
        y_ticks = np.arange(1, self.n + 1) * self.h
        z_ticks = np.arange(1, self.n + 1) * self.h
        
        y_mesh, z_mesh, x_mesh = np.meshgrid(x_ticks, y_ticks, z_ticks)

        x_flattened = x_mesh.flatten()
        y_flattened = y_mesh.flatten()
        z_flattened = z_mesh.flatten()

        self.b_3D = self.f(x_flattened, y_flattened, z_flattened)
        
        # z=0
        bc_facet_bottom = self.bc(x_flattened[:self.n*self.n], y_flattened[:self.n*self.n].flatten(), 0.)
        self.b_3D[:self.n*self.n] += bc_facet_bottom / (self.h**2)
        print(f"Z=0 bc correct: {np.all(z_flattened[:self.n*self.n]==0) and len(z_flattened[:self.n*self.n])==self.n**2}")

        # z=1
        bc_facet_top = self.bc(x_flattened[self.n**2*(self.n-1):], y_flattened[self.n**2*(self.n-1):], 1.)
        self.b_3D[self.n**2*(self.n-1):] += bc_facet_top / (self.h**2)
        print(f"Z=1 bc correct: {np.all(z_flattened[self.n**2*(self.n-1):]==1)}")

        idx_y_0 = np.array([range(k*self.n*self.n, k*self.n*self.n + self.n) for k in range(self.n)]).flatten()
        idx_y_1 = np.array([range(k*self.n*self.n-self.n, k*self.n*self.n) for k in range(self.n)]).flatten()
        idx_y_1 = np.array([range(k*self.n*self.n+self.n*(self.n-1), k*self.n*self.n + self.n*self.n) for k in range(self.n)]).flatten()
        
        # y=0
        bc_bottom = self.bc(x_flattened[idx_y_0], 0., z_flattened[idx_y_0])
        self.b_3D[idx_y_0] += bc_bottom / (self.h**2)
        print(f"Y=0 bc correct: {np.all(y_flattened[idx_y_0]==0)}")

        # y=1
        bc_top = self.bc(x_flattened[idx_y_1], 1., z_flattened[idx_y_1])
        self.b_3D[idx_y_1] += bc_top / (self.h**2)
        print(f"Y=1 bc correct: {np.all(y_flattened[idx_y_1]==1)}")

        # x=0
        bc_left = self.bc(0., y_flattened[::self.n], z_flattened[::self.n])
        self.b_3D[::self.n] += bc_left / (self.h**2)
        print(f"X=0 bc correct: {np.all(x_flattened[::self.n]==0)}")

        # Right boundary (x=1)
        bc_right = self.bc(1, y_flattened[self.n-1::self.n], z_flattened[self.n-1::self.n])
        self.b_3D[self.n-1::self.n] += bc_right / (self.h**2)
        print(f"X=1 bc correct: {np.all(x_flattened[self.n-1::self.n]==1)}")

    def solve(self):
        # Solve using numpy for verification
        # u = np.linalg.solve(self.A_3D, self.b_3D)
        # print("Solution vector u (numpy): \n", u)

        # Solve using custom solver
        u, cpu_time, peak_memory = self.solver.solve_banded()
        
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

        return u, e_rms, e_infty, cpu_time, peak_memory

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
