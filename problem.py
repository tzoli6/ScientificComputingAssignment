import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from solver import Solver


class PoissonProblem:

    def __init__(self, n):
        # self.domain = domain
        # self.boundary_conditions = boundary_conditions
        # self.solution = None
        self.n = n
        self.A_1D = None
        self.A_2D = None
        self.solver = Solver(self)

    def construct_1d_problem(self):
        A_i = -np.array([1, -2, 1])
        self.A_1D = np.diag(A_i[0]*np.ones(self.n-1), k=-1) + \
        np.diag(A_i[1]*np.ones(self.n), k=0) + \
        np.diag(A_i[2]*np.ones(self.n-1), k=1)
        self.A_1D /= (self.n)**2
        print("1D Problem Matrix A:", self.A_1D)

    def construct_2d_problem(self):
        if self.A_1D is None:
            print("1D problem not constructed yet. Constructing 1D problem...")
            self.construct_1d_problem()

        self.A_2D = np.kron(np.eye(self.n), self.A_1D) + np.kron(self.A_1D, np.eye(self.n))

        print("2D Problem Matrix A: \n", self.A_2D)

        print(f"A_2D is symmetric: {np.allclose(self.A_2D.T, self.A_2D)}")
        print(f"A_2D is positive definite: {np.all(np.linalg.eigvals(self.A_2D) > 0)}")
        print(np.linalg.eigvals(self.A_2D))
        self.solver.cholesky_decomposition()

    def construct_3d_problem(self):
        pass

    def solve(self):
        pass

    def plot_solution(self):
        pass

problem = PoissonProblem(n=2)
problem.construct_1d_problem()
problem.construct_2d_problem()