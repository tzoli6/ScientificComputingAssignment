
import numpy as np

class Solver:
    def __init__(self, problem):
        self.problem = problem

        self.C = None 

    def cholesky_decomposition(self):
        self.A_lower_triang = self.problem.A_2D.copy()
        C = np.zeros_like(self.A_lower_triang)
        for i in range(C.shape[0]):
            # print(f"Cholesky step {i+1}/{C.shape[0]}")
            C[i, i] = np.sqrt(self.A_lower_triang[i, i] - np.sum(C[i, :i]**2))
            self.A_lower_triang[i, i] = C[i, i]
            for j in range(i+1, C.shape[0]):
                C[j, i] = 1/C[i, i] * (self.A_lower_triang[j, i]-np.sum(C[j, :i]*C[i, :i]))
                self.A_lower_triang[j, i] = C[j, i]
        
        C = np.tril(self.A_lower_triang)
        print("Cholesky Decomposition Result (Lower Triangular Matrix): \n", C)
        print(np.round(C@C.T, 2))
        self.C = C
        return C
            

    def solve(self):
        # Forward substitution
        y = self.problem.b_2D.copy()
        for i in range(len(y)):
            for j in range(i):
                y[i] -= self.C[i, j] * y[j]
            y[i] /= self.C[i, i]

        # Backward substitution
        u = y.copy()
        C_T = self.C.T
        for i in range(len(u)-1, -1, -1):
            for j in range(i+1, len(u)):
                u[i] -= C_T[i, j] * u[j]
            u[i] /= C_T[i, i]

        y_np = np.linalg.solve(self.C, self.problem.b_2D)
        print(f"Forward step solution y: \n custom solver: {y} \n numpy: {y_np}")
        print("Solution vector u (custom solver): \n", u)
        return u.reshape((self.problem.n, self.problem.n))