"""Sparse deconvolution of continuous signals into spike-like events.

Treats an observed 1D trace as a sparse superposition of stereotyped kernels
(e.g. synaptic alpha shapes). Solves an L1-regularized linear inverse problem
to recover event timings/amplitudes—useful when exploring how filtered spike
trains relate to measured voltage or fluorescence-like signals.
"""

import numpy as np
from scipy.signal import fftconvolve
from sklearn.linear_model import Lasso


class SpikeDeconvolver:
    """Recover sparse event trains that explain an observed 1D signal.

    Models ``y ≈ sum_i (s_i * k_i)`` where each ``k_i`` is a fixed temporal
    kernel (alpha function) and ``s_i`` is a sparse coefficient sequence over
    time. Coefficients are fit with Lasso (optional nonnegativity), so the
    method prefers few events while reconstructing ``y``.

    Parameters
    ----------
    signal : array-like
        Observed 1D time series to deconvolve.
    dt : float, optional
        Time step between samples (arbitrary units). Default is 1.0.
    kernel_length : int, optional
        Number of samples in each kernel template. Default is 100.
    l1_lambda : float, optional
        Lasso sparsity penalty (``alpha`` in sklearn). Larger values yield
        fewer events. Default is 0.01.
    nonnegative : bool, optional
        If True, constrain coefficients to be nonnegative (excitatory-only
        events). Default is True.
    """

    def __init__(
        self, signal, dt=1.0, kernel_length=100, l1_lambda=0.01, nonnegative=True
    ):
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
        """Build a unit-norm alpha (synaptic) kernel with time constant ``tau``.

        Shape is ``(t/tau) * exp(1 - t/tau)`` on ``self.time_kernel``, clipped
        at zero and normalized. This is the stereotyped waveform each event
        is assumed to contribute to the observation.

        Parameters
        ----------
        tau : float
            Rise/decay time constant in the same units as ``dt``.

        Returns
        -------
        ndarray
            Kernel of length ``kernel_length``, L2-normalized.
        """
        t = self.time_kernel
        k = (t / tau) * np.exp(1 - t / tau)
        k[k < 0] = 0
        return k / (np.linalg.norm(k) + 1e-12)

    # ==========================================================
    # Build convolution matrix
    # ==========================================================

    def build_dictionary(self, kernels):
        """Construct a convolution dictionary for one or more kernels.

        For each kernel and each possible onset time, places a shifted copy of
        the kernel into a column. The resulting matrix ``D`` satisfies
        ``y ≈ D @ coeffs`` under the convolutional generative model.

        Parameters
        ----------
        kernels : list of ndarray
            Kernel waveforms (each length ``kernel_length``).

        Returns
        -------
        D : ndarray, shape (T, T * n_kernels)
            Dictionary matrix whose columns are time-shifted kernels.
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
        """Fit sparse coefficients for ``y ≈ D @ coeffs`` via Lasso.

        Parameters
        ----------
        D : ndarray
            Dictionary matrix from :meth:`build_dictionary`.

        Returns
        -------
        ndarray
            Coefficient vector (length ``D.shape[1]``). Nonnegative if
            ``self.nonnegative`` is True.
        """

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
        """Deconvolve with a single alpha kernel of time constant ``tau``.

        Builds the dictionary for one kernel, solves the sparse problem, and
        reconstructs the signal by convolving the recovered spike train with
        the kernel.

        Parameters
        ----------
        tau : float
            Alpha kernel time constant.

        Returns
        -------
        dict
            ``tau`` : float
                Kernel time constant used.
            ``kernel`` : ndarray
                Alpha kernel waveform.
            ``spike_train`` : ndarray, shape (T,)
                Recovered sparse event amplitudes over time.
            ``reconstruction`` : ndarray, shape (T,)
                Signal reconstructed from spike train ⊗ kernel.
            ``error`` : float
                L2 reconstruction error ``||y - y_hat||``.
        """

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
        """Deconvolve using several alpha kernels (one spike train per tau).

        Allows mixture of event shapes—e.g. fast and slow synaptic components—
        by stacking dictionaries for each ``tau`` and jointly fitting sparse
        coefficients.

        Parameters
        ----------
        taus : sequence of float
            Time constants for each alpha kernel.

        Returns
        -------
        dict
            ``taus`` : sequence
                Input time constants.
            ``kernels`` : list of ndarray
                Kernel waveforms.
            ``spike_trains`` : ndarray, shape (n_kernels, T)
                Sparse event trains, one row per kernel.
            ``reconstruction`` : ndarray, shape (T,)
                Sum of convolutions of each train with its kernel.
            ``error`` : float
                L2 reconstruction error.
        """

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
        """Choose the single-kernel ``tau`` that minimizes reconstruction error.

        Runs :meth:`fit_single` for each candidate and returns the best result.
        Useful when the true synaptic time constant is unknown.

        Parameters
        ----------
        tau_grid : sequence of float
            Candidate alpha time constants to search.

        Returns
        -------
        dict
            Same structure as :meth:`fit_single` for the lowest-error ``tau``.
        """

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
        """Coefficient of determination for a reconstruction of ``self.y``.

        Parameters
        ----------
        y_hat : ndarray
            Predicted / reconstructed signal, same length as ``self.y``.

        Returns
        -------
        float
            ``1 - SS_res / SS_tot``. Closer to 1 means a better fit.
        """
        return 1 - np.sum((self.y - y_hat) ** 2) / np.sum((self.y - self.y.mean()) ** 2)
