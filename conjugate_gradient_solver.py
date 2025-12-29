import scipy.sparse as sp
import numpy as np
import time
import tracemalloc

class ConjugateGradientSolver:
    def __init__(self, problem, tol=1e-10, max_iter=10000):
        """
        Initialize the Conjugate Gradient Solver.

        Parameters:
        A (scipy.sparse matrix): Symmetric positive-definite matrix.
        b (numpy array): Right-hand side vector.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.
        """
        self.tol = tol
        self.max_iter = max_iter
        self.error_history = []

        self.problem = problem

    def incomplete_cholesky_decomposition_banded(self):
        print("Starting incomplete Cholesky decomposition...")

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
        R = sp.lil_matrix((N, N))
        
        for i in range(N):
            i0 = max(0, i-bandwidth)

            Ci = C[i, i0:i].toarray()
            diag_val = self.A_lower_triang[i, i] - (Ci@Ci.T)[0, 0]
            
            if diag_val > 0:
                C_ii = np.sqrt(diag_val)
                C[i, i] = C_ii
            else:
                R[i, i] = diag_val
                continue

            j_end = min(N, i + bandwidth + 1)

            for j in range(i+1, j_end):
                if self.A_lower_triang[j, i] != 0:
                    C_ji = C[j, i0:i].toarray()
                    C_ji = (self.A_lower_triang[j, i] - (C_ji@Ci.T)[0, 0]) / C_ii
                    C[j, i] = C_ji
                else:
                    C_ji = C[j, i0:i].toarray()
                    R[j, i] = self.A_lower_triang[j, i] - (C_ji@Ci.T)[0, 0]
        
        self.C = C
        self.R = R

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Incomplete Cholesky completed in {cpu_time:.4f} seconds, peak memory usage: {peak / 10**6:.4f} MB")

        return C, R, cpu_time, peak
    
    def incomplete_cholesky_decomposition_banded_faster(self):
        print("Starting faster incomplete Cholesky decomposition...")

        tracemalloc.start()
        start_time = time.process_time()
        
        if self.problem.A_3D is not None:
            self.A_lower_triang = self.problem.A_3D.copy()
            bandwidth = self.problem.n**2
        else:
            self.A_lower_triang = self.problem.A_2D.copy()
            bandwidth = self.problem.n

        A_csr = self.A_lower_triang.tocsr()
        N = A_csr.shape[0]
        C = sp.lil_matrix((N, N))
        
        band_buffer = np.zeros(bandwidth + 1)
        
        for i in range(N):
            i0 = max(0, i-bandwidth)
            band_len = i - i0

            if band_len > 0:
                Ci = C[i, i0:i].toarray().ravel()
                band_buffer[:band_len] = Ci
                diag_sum = np.sum(band_buffer[:band_len]**2)
            else:
                diag_sum = 0.0
            
            diag_val = A_csr[i, i] - diag_sum
            
            if diag_val <= 0:
                continue
                
            C_ii = np.sqrt(diag_val)
            C[i, i] = C_ii
            C_ii_inv = 1.0 / C_ii

            j_end = min(N, i + bandwidth + 1)
            
            for j in range(i+1, j_end):
                A_ji = A_csr[j, i]
                if A_ji != 0:
                    if band_len > 0:
                        C_j = C[j, i0:i].toarray().ravel()
                        dot_product = np.dot(C_j, band_buffer[:band_len])
                    else:
                        dot_product = 0.0
                        
                    C[j, i] = (A_ji - dot_product) * C_ii_inv
        
        self.C = C.tocsr()

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Incomplete Cholesky completed in {cpu_time:.4f} seconds, peak memory usage: {peak / 10**6:.4f} MB")

        return self.C, cpu_time, peak
        
    def solve(self, A, b, M=None, u0=None):
        """
        Solve the linear system Ax = b using the Conjugate Gradient method.

        Parameters:
        x0 (numpy array): Initial guess for the solution.
        M  (scipy.sparse matrix): Preconditioner matrix.

        Returns:
        x  (numpy array): Approximate solution to the system.
        """

        tracemalloc.start()
        start_time = time.process_time()


        f_h_squared = b.T @ b
        print(f"Initial residual norm squared: {f_h_squared}")

        n = b.shape[0]

        if u0 is None:
            u0 = sp.csr_matrix((n, 1)).toarray().flatten()

        if M is None:
            # M = np.eye(A.shape[0]) 
            C, cpu_time, peak_memory = self.incomplete_cholesky_decomposition_banded_faster()
            M = C
        # 0 th iteration
        u = u0
        r = b - A @ u
        z = sp.linalg.spsolve(C, r)
        p = z.copy()

        for i in range(self.max_iter):

            alpha = r.dot(z) / p.dot(A @ p)
            u = u + alpha * p

            r1 = r - alpha*A @ p
            z1 = sp.linalg.spsolve(M, r1)

            beta = r1.dot(z1)/r.dot(z)
            p = z1 + beta*p

            r = r1
            z = z1

            if r.dot(r)/f_h_squared < self.tol**2:
                print(r.dot(r)/f_h_squared)
                print(f"Converged in {i+1} iterations.")
                break

            self.error_history.append(np.sqrt(r.dot(r)/f_h_squared))

        if i == self.max_iter - 1:
            print("Maximum iterations reached without convergence.")

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Conjugate Gradient completed in {cpu_time:.4f} seconds, peak memory usage: {peak / 10**6:.4f} MB")

        return u, cpu_time, peak