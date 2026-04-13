import numpy as np
from scipy.signal import fftconvolve
from sklearn.linear_model import Lasso


class SpikeDeconvolver:
    def __init__(
        self, signal, dt=1.0, kernel_length=100, l1_lambda=0.01, nonnegative=True
    ):
        """
        signal: 1D array (observed signal)
        dt: timestep
        kernel_length: support of kernel (in samples)
        l1_lambda: sparsity penalty
        """

        self.y = signal.astype(float)
        self.T = len(signal)
        self.dt = dt
        self.kernel_length = kernel_length
        self.l1_lambda = l1_lambda
        self.nonnegative = nonnegative

        self.time_kernel = np.arange(kernel_length) * dt

    # ==========================================================
    # Kernel definitions
    # ==========================================================

    def alpha_kernel(self, tau):
        t = self.time_kernel
        k = (t / tau) * np.exp(1 - t / tau)
        k[k < 0] = 0
        return k / (np.linalg.norm(k) + 1e-12)

    # ==========================================================
    # Build convolution matrix
    # ==========================================================

    def build_dictionary(self, kernels):
        """
        kernels: list of kernels
        returns dictionary matrix D (T x (T * num_kernels))
        """

        D_list = []

        for k in kernels:
            for shift in range(self.T):
                atom = np.zeros(self.T)
                end = min(self.T, shift + len(k))
                atom[shift:end] = k[: end - shift]
                D_list.append(atom)

        D = np.stack(D_list, axis=1)
        return D

    # ==========================================================
    # Solve sparse coding problem
    # ==========================================================

    def solve_sparse(self, D):

        model = Lasso(
            alpha=self.l1_lambda,
            fit_intercept=False,
            positive=self.nonnegative,
            max_iter=5000,
        )

        model.fit(D, self.y)
        return model.coef_

    # ==========================================================
    # Single kernel fit
    # ==========================================================

    def fit_single(self, tau):

        k = self.alpha_kernel(tau)
        D = self.build_dictionary([k])

        coeffs = self.solve_sparse(D)

        # reshape into spike train
        s = coeffs.reshape(1, self.T)[0]

        y_hat = fftconvolve(s, k, mode="full")[: self.T]

        return {
            "tau": tau,
            "kernel": k,
            "spike_train": s,
            "reconstruction": y_hat,
            "error": np.linalg.norm(self.y - y_hat),
        }

    # ==========================================================
    # Multi-kernel fit
    # ==========================================================

    def fit_multi(self, taus):

        kernels = [self.alpha_kernel(tau) for tau in taus]
        D = self.build_dictionary(kernels)

        coeffs = self.solve_sparse(D)

        num_k = len(taus)
        s = coeffs.reshape(num_k, self.T)

        y_hat = np.zeros(self.T)

        for i, k in enumerate(kernels):
            y_hat += fftconvolve(s[i], k, mode="full")[: self.T]

        return {
            "taus": taus,
            "kernels": kernels,
            "spike_trains": s,
            "reconstruction": y_hat,
            "error": np.linalg.norm(self.y - y_hat),
        }

    # ==========================================================
    # Grid search over tau (single kernel)
    # ==========================================================

    def fit_single_grid(self, tau_grid):

        best = None

        for tau in tau_grid:
            result = self.fit_single(tau)

            if best is None or result["error"] < best["error"]:
                best = result

        return best

    # ==========================================================
    # Utility: evaluate fit quality
    # ==========================================================

    def r2_score(self, y_hat):
        return 1 - np.sum((self.y - y_hat) ** 2) / np.sum((self.y - self.y.mean()) ** 2)
