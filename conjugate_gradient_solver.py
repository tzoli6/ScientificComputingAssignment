import scipy.sparse as sp
import numpy as np
import time
import tracemalloc
#from sksparse.cholmod import cholesky

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

    def incomplete_cholesky_decomposition_banded_faster(self):
        print("Starting faster incomplete Cholesky decomposition...")

        tracemalloc.start()
        start_time = time.process_time()

        if self.problem.A_3D is not None:
            A = self.problem.A_3D.tocsr()
        else:
            A = self.problem.A_2D.tocsr()

        N = A.shape[0]
        C = sp.lil_matrix((N, N))

        for i in range(N):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]

            # Diagonal
            diag = A[i, i]

            for idx, j in enumerate(cols):
                if j >= i:
                    continue

                lij = C[i, j]
                diag -= lij * lij

            if diag <= 0.0:
                raise RuntimeError(f"Non-positive pivot at row {i}")

            C_ii = np.sqrt(diag)
            C[i, i] = C_ii

            # Off-diagonals
            for idx, j in enumerate(cols):
                if j <= i:
                    continue

                aji = A[j, i]
                if aji == 0:
                    continue

                s = 0.0
                row_j_start = C.rows[j]
                row_j_data = C.data[j]

                for k_idx, k in enumerate(row_j_start):
                    if k >= i:
                        break
                    s += row_j_data[k_idx] * C[i, k]

                C[j, i] = (aji - s) / C_ii

        C = C.tocsr()

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(
            f"Incomplete Cholesky completed in {cpu_time:.4f} seconds, "
            f"peak memory usage: {peak / 10**6:.4f} MB"
        )

        return C, cpu_time, peak
    
    def cholesky_decomposition_cholmod(self, mode='supernodal'):
        print("Starting CHOLMOD Cholesky decomposition...")

        tracemalloc.start()
        start_time = time.process_time()

        if self.problem.A_3D is not None:
            A = self.problem.A_3D.tocsc()
        else:
            A = self.problem.A_2D.tocsc()

        # Compute Cholesky factorization
        factor = cholesky(A, mode=mode, ordering_method='best')
        L_perm = factor.L()
        P = factor.P()
    
        # Create permutation matrix
        P_inv = np.argsort(P)
        L_perm_csr = L_perm.tocsr()
        
        # Permute rows
        L_unperm = L_perm_csr[P_inv, :]
        # Permute columns
        L_unperm = L_unperm[:, P_inv]
        
        L = L_unperm.tocsr()

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"CHOLMOD Cholesky completed in {cpu_time:.4f} seconds, "f"peak memory: {peak / 10**6:.4f} MB.")

        return L, cpu_time, peak

    def solve_ic_bim(self, L=None, u0=None):
        """
        Solve A u = b using the incomplete Cholesky factorisation as a
        basic iterative method (defect–correction iteration).

        Parameters
        ----------
        L : scipy.sparse matrix or None
            Lower triangular incomplete Cholesky factor. If None, it is
            computed using incomplete_cholesky_decomposition_banded_faster().
        u0 : numpy.ndarray or None
            Initial guess. If None, the zero vector is used.

        Returns
        -------
        u : numpy.ndarray
            Approximate solution.
        cpu_time : float
            CPU time (seconds) for the iterative part.
        peak : float
            Peak memory usage (bytes) during the iterative phase.
        """
        # Select the correct system (2D vs 3D)
        if self.problem.A_3D is not None:
            A = self.problem.A_3D.tocsr()
            b = self.problem.b_3D
        else:
            A = self.problem.A_2D.tocsr()
            b = self.problem.b_2D

        n = b.shape[0]

        # Initial guess
        if u0 is None:
            u = np.zeros_like(b)
        else:
            u = u0.copy()

        # Denominator of relative residual
        f_h_squared = float(b.T @ b)

        # Build incomplete Cholesky factor if not supplied
        if L is None:
            L, _, _ = self.incomplete_cholesky_decomposition_banded_faster()
        else:
            if sp.issparse(L):
                L = L.tocsr()

        self.error_history = []

        tracemalloc.start()
        start_time = time.process_time()

        # Initial residual
        r = b - A @ u

        for k in range(self.max_iter):
            # Relative residual ||r||_2 / ||f_h||_2
            rel_res = np.sqrt(float(r.T @ r) / f_h_squared)
            self.error_history.append(rel_res)

            if rel_res <= self.tol:
                print(
                    f"IC-BIM converged in {k + 1} iterations with "
                    f"relative residual {rel_res:.3e}."
                )
                break

            # Solve M e = r with M = L L^T:
            #   L y = r
            #   L^T e = y
            y = sp.linalg.spsolve_triangular(L, r, lower=True)
            e = sp.linalg.spsolve_triangular(L.T, y, lower=False)

            # Update iterate and residual
            u = u + e
            r = b - A @ u

        else:
            print("IC-BIM: maximum number of iterations reached without convergence.")

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(
            f"IC-BIM iteration completed in {cpu_time:.4f} seconds, "
            f"peak memory usage: {peak / 10 ** 6:.4f} MB"
        )

        return u, cpu_time, peak

    def solve(self, M=None, u0=None):
        """
        Solve the linear system Ax = b using the Conjugate Gradient method.

        Parameters:
        x0 (numpy array): Initial guess for the solution.
        M  (scipy.sparse matrix): Preconditioner matrix.

        Returns:
        x  (numpy array): Approximate solution to the system.
        """
        if self.problem.A_3D is not None:
            A = self.problem.A_3D.tocsc()
            b = self.problem.b_3D
        else:
            A = self.problem.A_2D.tocsc()
            b = self.problem.b_2D

        f_h_squared = b.T @ b
        print(f"Initial residual norm squared: {f_h_squared}")

        n = b.shape[0]

        if u0 is None:
            u0 = sp.csr_matrix((n, 1)).toarray().flatten()

        if M is None:
            L, cpu_time, peak_memory = self.cholesky_decomposition_cholmod()

        tracemalloc.start()
        start_time = time.process_time()
        
        # 0 th iteration
        u = u0
        r = b - A @ u
        y = sp.linalg.spsolve_triangular(L, r, lower=True)
        z = sp.linalg.spsolve_triangular(L.T, y, lower=False)
        p = z.copy()

        for i in range(self.max_iter):

            alpha = r.dot(z) / p.dot(A @ p)
            u = u + alpha * p

            r1 = r - alpha*A @ p
            y1 = sp.linalg.spsolve_triangular(L, r1, lower=True)
            z1 = sp.linalg.spsolve_triangular(L.T, y1, lower=False)

            beta = r1.dot(z1)/r.dot(z)
            p = z1 + beta*p

            r = r1
            z = z1

            error = np.sqrt(r.dot(r)/f_h_squared)
            if error < self.tol**2:
                print(error)
                print(f"Converged in {i+1} iterations.")
                break

            self.error_history.append(error)

        if i == self.max_iter - 1:
            print("Maximum iterations reached without convergence.")

        cpu_time = time.process_time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Conjugate Gradient completed in {cpu_time:.4f} seconds, peak memory usage: {peak / 10**6:.4f} MB")

        return u, cpu_time, peak