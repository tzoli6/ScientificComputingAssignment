
import numpy as np

class Solver:
    def __init__(self, problem):
        self.problem = problem

    def cholesky_decomposition(self):
        self.A_lower_triang = self.problem.A_2D.copy()
        C = np.zeros_like(self.A_lower_triang)
        for i in range(self.problem.n):
            C[i, i] = np.sqrt(self.A_lower_triang[i, i] - np.sum(C[i, :i-1]**2)) # Check indexinf for the argument of sum
            self.A_lower_triang[i, i] = C[i, i].copy() # Check if there is reference
            for j in range(i+1, self.problem.n):
                C[j, i] = 1/C[i, i] * (self.A_lower_triang[j, i]-np.sum(C[j, :i]*C[i, :i-1]))
                self.A_lower_triang[j, i] = C[j, i].copy() # Check if there is reference)
        
        print("Cholesky Decomposition Result (Lower Triangular Matrix):", self.A_lower_triang)
            

    def solve(self):
        raise NotImplementedError("This method should be overridden by subclasses.")