
import numpy as np
import time
import tracemalloc
import scipy.sparse as sp
from sksparse.cholmod import cholesky
class Solver:
    def __init__(self, problem):
        self.problem = problem
        self.error_history = []

        self.C = None 

    def cholesky_decomposition_sparse(self):
        print("Starting sparse Cholesky decomposition...")

        tracemalloc.start()
        start_time = time.process_time()
        
        if self.problem.A_3D is not None:
            self.A_lower_triang = self.problem.A_3D.copy().tolil()
            bandwidth = self.problem.n**2
        else:
            self.A_lower_triang = self.problem.A_2D.copy().tolil()
            bandwidth = self.problem.n

        N = self.A_lower_triang.shape[0]
        C = sp.lil_matrix((N, N))
        
        for i in range(N):
            i0 = max(0, i-bandwidth)

            # Chace row
            Ci = C[i, i0:i].toarray()
            C_ii = np.sqrt(self.A_lower_triang[i, i] - (Ci@Ci.T)[0, 0])
            C[i, i] = C_ii

            j_end = min(N, i + bandwidth + 1)

            for j in range(i+1, j_end):
                C_ji = C[j, i0:i].toarray()
                C_ji = (self.A_lower_triang[j, i] - (C_ji@Ci.T)[0, 0]) / C_ii
                C[j, i] = C_ji
    
        self.C = C

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Sparse Cholesky decomposition completed in {cpu_time:.4f} seconds, peak memory usage: {peak / 10**6:.4f} MB")

        return self.C, cpu_time, peak
    
    def cholesky_decomposition_dense(self):
        tracemalloc.start()
        start_time = time.process_time()
        if self.problem.A_3D is not None:
            self.A_lower_triang = self.problem.A_3D.copy()
        else:
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
        # print("Cholesky Decomposition Result (Lower Triangular Matrix): \n", C)
        # print(np.round(C@C.T, 2))
        self.C = C

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return C, cpu_time, peak

    def cholesky_decomposition_scipy(self):
        
        if self.problem.A_3D is not None:
            self.A_lower_triang = self.problem.A_3D.copy()
        else:
            self.A_lower_triang = self.problem.A_2D.copy()
        
        tracemalloc.start()
        start_time = time.process_time()

        factor = cholesky(self.A_lower_triang, mode="simplicial")

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.C = factor.L()
        self.perm = factor.P()
        self.inv_perm = np.argsort(self.perm)


        print(f"SciPy Cholesky decomposition completed in {cpu_time:.4f} seconds, peak memory usage: {peak / 10**6:.4f} MB")

        return self.C, cpu_time, peak  

    def cholesky_decomposition_sparse(self):
        print("Starting sparse Cholesky decomposition...")

        tracemalloc.start()
        start_time = time.process_time()
        
        if self.problem.A_3D is not None:
            self.A_lower_triang = self.problem.A_3D.copy().tolil()
            bandwidth = self.problem.n**2
        else:
            self.A_lower_triang = self.problem.A_2D.copy().tolil()
            bandwidth = self.problem.n

        N = self.A_lower_triang.shape[0]
        C = sp.lil_matrix((N, N))
        
        for i in range(N):
            i0 = max(0, i-bandwidth)
            band_len = i - i0

            # Cache row and avoid repeated matrix multiplications
            if band_len > 0:
                Ci = C[i, i0:i].toarray().ravel()  # Flatten to 1D
                C_ii = np.sqrt(self.A_lower_triang[i, i] - np.dot(Ci, Ci))  # Use dot instead of @
            else:
                C_ii = np.sqrt(self.A_lower_triang[i, i])
                
            C[i, i] = C_ii
            C_ii_inv = 1.0 / C_ii  # Precompute inverse

            j_end = min(N, i + bandwidth + 1)

            for j in range(i+1, j_end):
                if band_len > 0:
                    C_ji = C[j, i0:i].toarray().ravel()  # Flatten to 1D
                    C[j, i] = (self.A_lower_triang[j, i] - np.dot(C_ji, Ci)) * C_ii_inv  # Reuse Ci
                else:
                    C[j, i] = self.A_lower_triang[j, i] * C_ii_inv
        
        self.C = C.tocsr()  # Convert to CSR at end

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Sparse Cholesky completed in {cpu_time:.4f} seconds, peak memory: {peak / 10**6:.4f} MB")

        return self.C, cpu_time, peak

    def solve_dense(self):
        pass

    def solve_scipy(self):
        print("Starting SciPy solver...")

        if self.problem.b_3D is not None:
            b = self.problem.b_3D
        else:
            b = self.problem.b_2D

        tracemalloc.start()
        start_time = time.process_time()

        b_perm = b[self.perm]
        # Forwad substitution
        y = sp.linalg.spsolve_triangular(self.C, b_perm, lower=True)
        # Backward substitution
        z = sp.linalg.spsolve_triangular(self.C.T,  y, lower=False)

        u = z[self.inv_perm]

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"SciPy solver completed in {cpu_time:.4f} seconds, peak memory usage: {peak / 10**6:.4f} MB")

        return u, cpu_time, peak
    
    def solve_banded(self):
        print("Starting banded solver...")

        tracemalloc.start()
        start_time = time.process_time()

        if sp.issparse(self.C):
            self.C = self.C.tocsr()

        # Forward substitution
        if self.problem.b_3D is not None:
            y = self.problem.b_3D.copy()
            bandwidth = self.problem.n**2
        else:
            y = self.problem.b_2D.copy()
            bandwidth = self.problem.n

        for i in range(len(y)):
            i_start = max(0, i-bandwidth)
            # ONLY CHANGE: extract row once, use np.dot
            if i > i_start:
                C_row = self.C[i, i_start:i].toarray().ravel()
                y[i] = (y[i] - np.dot(C_row, y[i_start:i])) / self.C[i, i]
            else:
                y[i] /= self.C[i, i]

        # Backward substitution
        u = y.copy()
        C_T = self.C.T
        if sp.issparse(C_T):
            C_T = C_T.tocsr()
        
        for i in range(len(u)-1, -1, -1):
            i_end = min(len(u), i + bandwidth + 1)
            # ONLY CHANGE: extract row once, use np.dot
            if i_end > i + 1:
                C_T_row = C_T[i, i+1:i_end].toarray().ravel()
                u[i] = (u[i] - np.dot(C_T_row, u[i+1:i_end])) / C_T[i, i]
            else:
                u[i] /= C_T[i, i]

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Banded solver completed in {cpu_time:.4f} seconds, peak memory usage: {peak / 10**6:.4f} MB")

        return u, cpu_time, peak

    def solve(self):
        tracemalloc.start()
        start_time = time.process_time()

        # Forward substitution
        if self.problem.b_3D is not None:
            y = self.problem.b_3D.copy()
        else:
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

        # y_np = np.linalg.solve(self.C, self.problem.b_3D)
        # print(f"Forward step solution y: \n custom solver: {y} \n numpy: {y_np}")
        # print("Solution vector u (custom solver): \n", u)

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return u, cpu_time, peak