"""Synaptic integration toy model and batch signal-complexity analyses.

``Integrator`` simulates multi-synapse dendritic/voltage responses with
optional saturation nonlinearity, for exploring how morphology filters and
input statistics shape temporal structure. ``BatchSignalAnalyzer`` and
``NonnegativeBatchSignalAnalyzer`` compare two trial×time ensembles with
entropy, dimensionality, compression, and NMF-based metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.decomposition import PCA, NMF
from sklearn.neighbors import NearestNeighbors
from scipy.fft import rfft
import zlib
from tqdm import tqdm


def make_epoch_times(total_time=500.0, n_epochs=3, buffer_fraction=0.1):
    """Partition a trial into contiguous active epochs with edge buffers.

    Leaves a fraction of ``total_time`` quiet at the start and end, then splits
    the remaining middle into ``n_epochs`` equal intervals. Used by
    :class:`Integrator` so Poisson drive is piecewise-stationary within epochs
    rather than spanning the entire trial.

    Parameters
    ----------
    total_time : float, optional
        Full trial duration (same units as returned edges; typically ms).
        Default is 500.0.
    n_epochs : int, optional
        Number of equal-length active epochs. Default is 3.
    buffer_fraction : float, optional
        Fraction of ``total_time`` reserved as buffer on each side.
        Default is 0.1 (10% start + 10% end).

    Returns
    -------
    list of tuple of float
        ``(start, end)`` pairs for each epoch, length ``n_epochs``.
    """
    buffer_time = total_time * buffer_fraction
    epoch_time = (total_time - 2 * buffer_time) / n_epochs
    return [
        (buffer_time + i * epoch_time, buffer_time + (i + 1) * epoch_time)
        for i in range(n_epochs)
    ]


@dataclass
class IntegratorOptions:
    """Configuration for :class:`Integrator` simulations.

    Groups timing, synapse count/rate statistics, morphology filter ranges,
    Dirichlet rate allocation, and nonlinearity into one object so experiments
    can be varied without long argument lists.

    Attributes
    ----------
    total_time_ms : float
        Trial duration in ms.
    time_step_ms : float
        Simulation time step in ms.
    n_trials : int
        Number of trials in :meth:`Integrator.run_trials`.
    n_epochs : int
        Number of active rate epochs per trial.
    buffer_fraction : float
        Quiet fraction at each end of the trial (see :func:`make_epoch_times`).
    n_synapses : int
        Number of input synapses / pathways.
    synapse_tau_ms : float
        Alpha (AMPA-like) synaptic time constant in ms.
    total_rate_hz : float
        Total Poisson rate (Hz) distributed across active synapses per epoch.
    active_synapse_fraction : float or str
        Fraction of synapses active each trial, or ``'random'`` to draw the
        count uniformly each call to rate generation.
    morphology_tau_range : tuple of float
        ``(min, max)`` exponential morphology filter time constants (ms).
    morphology_amplitude_range : tuple of float
        ``(min, max)`` synaptic amplitude scales.
    sampling_method : str
        How to assign morphology params: ``'uniform'`` or ``'linspace'``.
    dirichlet_alpha : float
        Concentration for Dirichlet weights that split ``total_rate_hz`` across
        active synapses (smaller → more uneven allocation).
    nonlinear : bool
        If True, apply a saturating transform to the summed voltage.
    random_seed : int
        Seed for the integrator RNG.
    """

    total_time_ms: float = 500.0
    time_step_ms: float = 0.1
    n_trials: int = 100
    n_epochs: int = 3
    buffer_fraction: float = 0.1

    n_synapses: int = 1
    synapse_tau_ms: float = 2.0  # AMPA alpha time constant
    total_rate_hz: float = 20.0  # total rate distributed across synapses
    active_synapse_fraction: float | str = (
        1.0  # fraction of synapses randomly chosen to be active per trial. enter 'random' to choose randomly each trial
    )

    morphology_tau_range: tuple[float, float] = (
        2.0,
        2.1,
    )  # morphology filter time constants
    morphology_amplitude_range: tuple[float, float] = (
        1.0,
        1.1,
    )  # morphology filter amplitudes
    sampling_method: str = "uniform"

    dirichlet_alpha: float = 0.5  # variance of rate distribution over active synapses
    nonlinear: bool = True
    random_seed: int = 0


class Integrator:
    """Multi-synapse filtered Poisson drive → voltage-like trial traces.

    Generative picture per trial:

    1. Choose a subset of active synapses and, within each epoch, allocate
       ``total_rate_hz`` across them with a Dirichlet draw.
    2. Draw independent Poisson spike trains from those piecewise rates.
    3. Filter each train with synapse alpha ⊗ morphology exponential, scale
       by amplitude, and sum.
    4. Optionally apply a saturating nonlinearity.

    Morphological diversity (τ and amplitude ranges) controls how heterogeneous
    the effective kernels are—central for studying temporal integration and
    signal complexity across conditions.

    Parameters
    ----------
    options : IntegratorOptions
        Full simulation configuration.
    """

    def __init__(self, options: IntegratorOptions):
        """Build kernels, sample morphology parameters, and cache time axes.

        Parameters
        ----------
        options : IntegratorOptions
            Simulation settings (seed, rates, morphology ranges, etc.).
        """
        self.opts = options
        self.rng = np.random.default_rng(self.opts.random_seed)
        self.n_time_steps = int(self.opts.total_time_ms / self.opts.time_step_ms)
        self.time = np.arange(self.n_time_steps) * self.opts.time_step_ms
        self.kernel_time = np.arange(0, 50, self.opts.time_step_ms)
        self.alpha_kernel = self._create_alpha_kernel()
        self.morphology_amplitudes = self._sample_amplitudes(
            method=self.opts.sampling_method
        )
        self.morphology_time_constants = self._sample_time_contants(
            method=self.opts.sampling_method
        )
        self.epoch_times = make_epoch_times(
            total_time=self.opts.total_time_ms,
            n_epochs=self.opts.n_epochs,
            buffer_fraction=self.opts.buffer_fraction,
        )

    # alpha synapse kernel
    def _create_alpha_kernel(self):
        """Return a peak-normalized alpha synaptic kernel on ``kernel_time``.

        Returns
        -------
        ndarray
            Alpha waveform ``(t/τ) exp(1 - t/τ)`` with τ = ``synapse_tau_ms``.
        """
        t = self.kernel_time
        tau = self.opts.synapse_tau_ms
        kernel = (t / tau) * np.exp(1 - t / tau)
        kernel /= np.max(kernel)
        return kernel

    # morphology kernel
    def _create_morphology_kernel(self, tau_ms):
        """Return a unit-peak exponential dendritic/morphology filter.

        Parameters
        ----------
        tau_ms : float
            Decay time constant in ms.

        Returns
        -------
        ndarray
            ``exp(-t / tau_ms)`` on ``kernel_time``.
        """
        return np.exp(-self.kernel_time / tau_ms)

    def _create_combined_kernel(self, tau_ms):
        """Convolve synaptic alpha with morphology exponential for one synapse.

        Parameters
        ----------
        tau_ms : float
            Morphology time constant in ms.

        Returns
        -------
        ndarray
            Combined kernel truncated to ``len(kernel_time)``.
        """
        morph_kernel = self._create_morphology_kernel(tau_ms)
        combined = np.convolve(self.alpha_kernel, morph_kernel)
        return combined[: len(self.kernel_time)]

    # Input generation
    def _generate_rate_matrix(self):
        """Sample piecewise-constant Poisson rates for all synapses.

        Within each epoch, active synapses share ``total_rate_hz`` via a
        Dirichlet draw (concentration ``dirichlet_alpha``). Inactive synapses
        remain at zero.

        Returns
        -------
        ndarray, shape (n_synapses, n_time_steps)
            Instantaneous rates in Hz.
        """
        rates = np.zeros((self.opts.n_synapses, self.n_time_steps))
        if self.opts.active_synapse_fraction == "random":
            n_active = self.rng.integers(low=1, high=self.opts.n_synapses)
        else:
            n_active = int(self.opts.active_synapse_fraction * self.opts.n_synapses)
        active_syn_indices = self.rng.choice(
            self.opts.n_synapses,
            size=n_active,
            replace=False,
        )
        for start_ms, end_ms in self.epoch_times:
            start_idx = int(start_ms / self.opts.time_step_ms)
            end_idx = int(end_ms / self.opts.time_step_ms)
            weights = self.rng.dirichlet(np.ones(n_active) * self.opts.dirichlet_alpha)
            rates[active_syn_indices, start_idx:end_idx] = (
                weights[:, None] * self.opts.total_rate_hz
            )
        return rates

    def _generate_poisson_spikes(self, rate_trace):
        """Bernoulli approximation to an inhomogeneous Poisson spike train.

        Parameters
        ----------
        rate_trace : ndarray
            Rate in Hz at each time step.

        Returns
        -------
        ndarray
            Binary spike indicators (0/1) of the same length.
        """
        probability = rate_trace * self.opts.time_step_ms / 1000.0
        return (self.rng.random(len(rate_trace)) < probability).astype(float)

    def _sample_amplitudes(self, method="uniform"):
        """Sample per-synapse morphology amplitudes from the configured range.

        Parameters
        ----------
        method : {'uniform', 'linspace'}, optional
            Random draws vs evenly spaced values across synapses.

        Returns
        -------
        ndarray, shape (n_synapses,)
            Amplitude scales.
        """
        a0, a1 = self.opts.morphology_amplitude_range
        if method == "uniform":
            samples = self.rng.uniform(low=a0, high=a1, size=self.opts.n_synapses)
        elif method == "linspace":
            samples = np.linspace(a0, a1, self.opts.n_synapses)
        else:
            raise ValueError(f"Unknown amplitude sampling method: {method}")
        return samples

    def _sample_time_contants(self, method="uniform"):
        """Sample per-synapse morphology time constants from the configured range.

        Parameters
        ----------
        method : {'uniform', 'linspace'}, optional
            Random draws vs evenly spaced values across synapses.

        Returns
        -------
        ndarray, shape (n_synapses,)
            Morphology τ values in ms.
        """
        t0, t1 = self.opts.morphology_tau_range
        if method == "uniform":
            samples = self.rng.uniform(low=t0, high=t1, size=self.opts.n_synapses)
        elif method == "linspace":
            samples = np.linspace(t0, t1, self.opts.n_synapses)
        else:
            raise ValueError(f"Unknown amplitude sampling method: {method}")
        return samples

    # Simulation core
    def simulate_single_trial(self):
        """Simulate one summed voltage-like trace from filtered synaptic spikes.

        Returns
        -------
        ndarray, shape (n_time_steps,)
            Trial waveform (linear sum, optionally nonlinearly saturated).
        """
        amplitudes = self.morphology_amplitudes
        tau_filters = self.morphology_time_constants
        rate_matrix = self._generate_rate_matrix()
        output_voltage = np.zeros(self.n_time_steps)
        for syn_index in range(self.opts.n_synapses):
            spike_train = self._generate_poisson_spikes(rate_matrix[syn_index])
            combined_kernel = self._create_combined_kernel(tau_filters[syn_index])
            filtered_signal = np.convolve(spike_train, combined_kernel, mode="full")[
                : self.n_time_steps
            ]
            output_voltage += amplitudes[syn_index] * filtered_signal
        if self.opts.nonlinear:
            output_voltage = output_voltage / (1 + 0.02 * np.abs(output_voltage))
        return output_voltage

    def run_trials(self, per_trial_normalization=None):
        """Simulate many trials and optionally normalize each waveform.

        Retries a trial if its L2 norm is near zero (empty/silent draws).

        Parameters
        ----------
        per_trial_normalization : str or None, optional
            None (raw), ``'divide_by_norm'``, ``'divide_by_max'``,
            ``'divide_by_mean'``, or ``'unit_gaussian'`` (z-score each trial).

        Returns
        -------
        ndarray, shape (n_trials, n_time_steps)
            Trial ensemble.
        """
        results = np.zeros((self.opts.n_trials, self.n_time_steps))
        for trial_index in tqdm(range(self.opts.n_trials)):
            while np.linalg.norm(results[trial_index]) < 1e-6:
                results[trial_index] = self.simulate_single_trial()
        if per_trial_normalization is None:
            pass
        elif per_trial_normalization == "divide_by_norm":
            results /= np.linalg.norm(results, axis=1)[:, None]
        elif per_trial_normalization == "divide_by_max":
            results /= results.max(axis=1)[:, None]
        elif per_trial_normalization == "divide_by_mean":
            results /= results.mean(axis=1)[:, None]
        elif per_trial_normalization == "unit_gaussian":
            centered = results - results.mean(axis=1)[:, None]
            results = centered / np.std(centered, axis=1)[:, None]
        else:
            raise ValueError(f"Unknown normalization method: {per_trial_normalization}")
        return results

    # -------------------------------------------------------
    # Analysis
    # -------------------------------------------------------

    def compute_participation_ratio(self, data_matrix):
        """Participation ratio of a trial×feature matrix via singular values.

        ``PR = (sum s_i)^2 / sum s_i^2`` after centering columns. Higher values
        indicate variance spread across more modes (effective dimensionality).

        Parameters
        ----------
        data_matrix : ndarray
            Typically shape ``(n_trials, n_time_steps)``.

        Returns
        -------
        participation_ratio : float
            Scalar effective dimensionality.
        singular_values : ndarray
            Singular values of the centered matrix.
        """
        centered = data_matrix - data_matrix.mean(axis=0)
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        participation_ratio = (np.sum(singular_values) ** 2) / np.sum(
            singular_values**2
        )
        return participation_ratio, singular_values


class BatchSignalAnalyzer:
    """Compare complexity / geometry of two centered trial×time ensembles.

    Holds conditions ``Y_A`` and ``Y_B`` (same shape), column-centers them, and
    exposes paired metrics: kNN entropy, spectral (PCA) entropy, compression,
    intrinsic dimension, pairwise distances, and power-spectrum entropy.
    Intended for asking whether one generative regime produces richer or more
    compressible temporal signals than another.

    Parameters
    ----------
    Y_A, Y_B : ndarray, shape (n_trials, n_timepoints)
        Trial ensembles to compare (must match in shape).
    """

    def __init__(self, Y_A, Y_B):
        """Store copies of both ensembles and subtract per-timepoint means.

        Parameters
        ----------
        Y_A, Y_B : ndarray, shape (n_trials, n_timepoints)
            Input trial matrices (identical shapes required).
        """
        assert Y_A.shape == Y_B.shape, "Inputs must have same shape"
        self.Y_A = Y_A.copy()
        self.Y_B = Y_B.copy()

        # center signals (recommended baseline preprocessing)
        self.Y_A -= self.Y_A.mean(axis=0, keepdims=True)
        self.Y_B -= self.Y_B.mean(axis=0, keepdims=True)

    # ==========================================================
    # 1. kNN entropy (Kozachenko–Leonenko style)
    # ==========================================================

    def knn_entropy(self, Y, k=5):
        """Proxy for differential entropy via mean log kNN distance.

        Larger values suggest a more spread / diverse cloud of trial vectors
        in time-series space (Kozachenko–Leonenko-style estimator, up to
        additive constants not computed here).

        Parameters
        ----------
        Y : ndarray, shape (n_trials, n_features)
            Points (trials) in feature space.
        k : int, optional
            Neighbor index used for the distance (excluding self). Default 5.

        Returns
        -------
        float
            Mean ``log(r_k + eps)`` over trials.
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Y)
        distances, _ = nbrs.kneighbors(Y)
        r = distances[:, -1]
        return np.mean(np.log(r + 1e-12))

    def compare_knn_entropy(self, k=5):
        """kNN entropy for conditions A and B.

        Parameters
        ----------
        k : int, optional
            Neighbor count for the estimator. Default is 5.

        Returns
        -------
        dict
            ``{'A': float, 'B': float}``.
        """
        return {"A": self.knn_entropy(self.Y_A, k), "B": self.knn_entropy(self.Y_B, k)}

    # ==========================================================
    # 2. Spectral entropy (PCA-based)
    # ==========================================================

    def spectral_entropy(self, Y):
        """Entropy of normalized singular-value power (PCA spectrum flatness).

        Higher values mean energy is distributed across more modes rather than
        concentrated in a few principal components.

        Parameters
        ----------
        Y : ndarray
            Trial×feature matrix.

        Returns
        -------
        float
            ``-sum p log p`` with ``p`` proportional to ``s_i^2``.
        """
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        power = S**2
        p = power / np.sum(power)
        return -np.sum(p * np.log(p + 1e-12))

    def compare_spectral_entropy(self):
        """Spectral entropy for conditions A and B.

        Returns
        -------
        dict
            ``{'A': float, 'B': float}``.
        """
        return {
            "A": self.spectral_entropy(self.Y_A),
            "B": self.spectral_entropy(self.Y_B),
        }

    # ==========================================================
    # 3. Compression ratio
    # ==========================================================

    def compression_ratio(self, Y):
        """Mean zlib compressed/raw size after 8-bit scaling of each trial.

        Higher ratio ⇒ less compressible ⇒ more algorithmic complexity / less
        repetitive temporal structure (heuristic).

        Parameters
        ----------
        Y : ndarray, shape (n_trials, n_timepoints)
            Trial waveforms.

        Returns
        -------
        float
            Average compression ratio across trials.
        """
        ratios = []
        for trial in Y:
            # normalize to 8-bit for stable compression
            scaled = (trial - trial.min()) / (np.ptp(trial) + 1e-12)
            bytes_data = (scaled * 255).astype(np.uint8).tobytes()
            compressed = zlib.compress(bytes_data)
            ratios.append(len(compressed) / len(bytes_data))
        return np.mean(ratios)

    def compare_compression(self):
        """Compression ratios for conditions A and B.

        Returns
        -------
        dict
            ``{'A': float, 'B': float}``.
        """
        return {
            "A": self.compression_ratio(self.Y_A),
            "B": self.compression_ratio(self.Y_B),
        }

    # ==========================================================
    # 4. Intrinsic dimensionality
    # ==========================================================

    def participation_ratio(self, Y):
        """SVD participation ratio of ``Y`` (no extra centering).

        Parameters
        ----------
        Y : ndarray
            Trial×feature matrix (already centered in ``__init__`` for A/B).

        Returns
        -------
        float
            ``(sum s)^2 / sum s^2``.
        """
        _, S, _ = np.linalg.svd(Y, full_matrices=False)
        return (np.sum(S) ** 2) / np.sum(S**2)

    def levina_bickel_dimension(self, Y, k=10):
        """Local intrinsic dimension via the Levina–Bickel MLE.

        Estimates manifold dimension from ratios of kNN distances in trial
        space—complementary to global participation ratio.

        Parameters
        ----------
        Y : ndarray, shape (n_trials, n_features)
            Points to embed.
        k : int, optional
            Number of neighbors (excluding self). Default is 10.

        Returns
        -------
        float
            Mean local dimension estimate over trials.
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Y)
        distances, _ = nbrs.kneighbors(Y)

        d = []
        for i in range(Y.shape[0]):
            r = distances[i, 1:]  # skip self-distance
            logs = np.log(r[-1] / (r[:-1] + 1e-12))
            d.append((len(r) - 1) / np.sum(logs))
        return np.mean(d)

    def compare_intrinsic_dimension(self):
        """Participation ratio and Levina–Bickel dimension for A and B.

        Returns
        -------
        dict
            Keys ``PR_A``, ``PR_B``, ``LB_A``, ``LB_B``.
        """
        return {
            "PR_A": self.participation_ratio(self.Y_A),
            "PR_B": self.participation_ratio(self.Y_B),
            "LB_A": self.levina_bickel_dimension(self.Y_A),
            "LB_B": self.levina_bickel_dimension(self.Y_B),
        }

    # ==========================================================
    # 5. Pairwise distance statistics
    # ==========================================================

    def pairwise_distances(self, Y):
        """Mean and std of pairwise Euclidean distances between trials.

        Summarizes how spread out trial waveforms are in time-series space.

        Parameters
        ----------
        Y : ndarray, shape (n_trials, n_features)
            Trial vectors.

        Returns
        -------
        dict
            ``{'mean': float, 'std': float}``.
        """
        from scipy.spatial.distance import pdist

        d = pdist(Y, metric="euclidean")
        return {"mean": np.mean(d), "std": np.std(d)}

    def compare_distances(self):
        """Pairwise distance stats for conditions A and B.

        Returns
        -------
        dict
            ``{'A': {...}, 'B': {...}}`` with mean/std each.
        """
        return {
            "A": self.pairwise_distances(self.Y_A),
            "B": self.pairwise_distances(self.Y_B),
        }

    # ==========================================================
    # 6. Power spectrum entropy
    # ==========================================================

    def power_spectrum_entropy(self, Y):
        """Mean Shannon entropy of each trial's normalized FFT power spectrum.

        Higher values indicate richer / flatter frequency content across trials.

        Parameters
        ----------
        Y : ndarray, shape (n_trials, n_timepoints)
            Trial waveforms.

        Returns
        -------
        float
            Average spectral entropy over trials.
        """
        entropies = []
        for trial in Y:
            spectrum = np.abs(rfft(trial)) ** 2
            p = spectrum / np.sum(spectrum)
            entropies.append(-np.sum(p * np.log(p + 1e-12)))
        return np.mean(entropies)

    def compare_power_entropy(self):
        """Power-spectrum entropy for conditions A and B.

        Returns
        -------
        dict
            ``{'A': float, 'B': float}``.
        """
        return {
            "A": self.power_spectrum_entropy(self.Y_A),
            "B": self.power_spectrum_entropy(self.Y_B),
        }

    # ==========================================================
    # Convenience: run all analyses
    # ==========================================================

    def run_all(self):
        """Run all pairwise comparison metrics and return a nested dict.

        Returns
        -------
        dict
            Keys: ``knn_entropy``, ``spectral_entropy``, ``compression``,
            ``intrinsic_dimension``, ``pairwise_distances``, ``power_entropy``.
        """
        return {
            "knn_entropy": self.compare_knn_entropy(),
            "spectral_entropy": self.compare_spectral_entropy(),
            "compression": self.compare_compression(),
            "intrinsic_dimension": self.compare_intrinsic_dimension(),
            "pairwise_distances": self.compare_distances(),
            "power_entropy": self.compare_power_entropy(),
        }

    def run_all_print(self):
        """Compute :meth:`run_all` and print human-readable interpretations.

        Returns
        -------
        None
        """
        results = self.run_all()
        print("\n--- kNN Entropy:")
        print("\tdistribution complexity. higher -> more diverse signals")
        print(f"A: {results['knn_entropy']['A']:.3f}")
        print(f"B: {results['knn_entropy']['B']:.3f}")
        print("\n--- Spectral Entropy:")
        print("\tsimilar to PCA. higher -> more balanced modes")
        print(f"A: {results['spectral_entropy']['A']:.3f}")
        print(f"B: {results['spectral_entropy']['B']:.3f}")
        print("\n--- Compression Ratio:")
        print("\talgorithmic complexity. higher -> less compressible, more complex")
        print(f"A: {results['compression']['A']:.3f}")
        print(f"B: {results['compression']['B']:.3f}")
        print("\n--- Intrinsic Dimension:")
        print("\tgeometric richness. higher -> larger manifold representation")
        print(f"Part. Ratio A: {results['intrinsic_dimension']['PR_A']:.3f}")
        print(f"Part. Ratio B: {results['intrinsic_dimension']['PR_B']:.3f}")
        print(f"LB Dimension A: {results['intrinsic_dimension']['LB_A']:.3f}")
        print(f"LB Dimension B: {results['intrinsic_dimension']['LB_B']:.3f}")
        print("\n--- Pairwise Distances:")
        print("\tsignal diversity. higher -> more spread out")
        print(f"Mean A: {results['pairwise_distances']['A']['mean']:.3f}")
        print(f"Std A: {results['pairwise_distances']['A']['std']:.3f}")
        print(f"Mean B: {results['pairwise_distances']['B']['mean']:.3f}")
        print(f"Std B: {results['pairwise_distances']['B']['std']:.3f}")
        print("\n--- Power Spectrum Entropy:")
        print("\ttemporal complexity -> richer frequency content")
        print(f"A: {results['power_entropy']['A']:.3f}")
        print(f"B: {results['power_entropy']['B']:.3f}")
        print("\n")


class NonnegativeBatchSignalAnalyzer:
    """Compare two nonnegative trial×time ensembles with NMF-aware metrics.

    Unlike :class:`BatchSignalAnalyzer`, preserves nonnegativity (optional
    per-trial max normalization) and emphasizes additive parts-based structure
    via NMF participation ratio, reconstruction curves, and Hoyer sparsity,
    plus kNN entropy, compression, and PCA PR as a reference.

    Parameters
    ----------
    Y_A, Y_B : ndarray, shape (n_trials, n_timepoints)
        Nonnegative trial ensembles (same shape).
    normalize : bool, optional
        If True, shift/scale each trial to ``[0, 1]``. Default is True.
    """

    def __init__(self, Y_A, Y_B, normalize=True):
        """Copy ensembles and optionally normalize each trial to unit max.

        Parameters
        ----------
        Y_A, Y_B : ndarray, shape (n_trials, n_timepoints)
            Nonnegative trial matrices.
        normalize : bool, optional
            Apply :meth:`_normalize` to both. Default is True.
        """

        assert Y_A.shape == Y_B.shape

        self.Y_A = Y_A.copy()
        self.Y_B = Y_B.copy()

        if normalize:
            self.Y_A = self._normalize(self.Y_A)
            self.Y_B = self._normalize(self.Y_B)

    # ==========================================================
    # Preprocessing
    # ==========================================================

    def _normalize(self, Y):
        """Shift each trial to min 0 and scale so max is 1.

        Parameters
        ----------
        Y : ndarray, shape (n_trials, n_timepoints)
            Input ensemble.

        Returns
        -------
        ndarray
            Per-trial normalized copy.
        """
        Y = Y - Y.min(axis=1, keepdims=True)
        Y = Y / (Y.max(axis=1, keepdims=True) + 1e-12)
        return Y

    # ==========================================================
    # Unified NMF Analysis
    # ==========================================================

    def _hoyer_sparsity(self, x):
        """Hoyer sparsity of a vector (0 = dense, 1 = maximally sparse).

        Parameters
        ----------
        x : ndarray
            Nonnegative (or absolute-valued in L1/L2) component loadings.

        Returns
        -------
        float
            Sparsity in ``[0, 1]``.
        """
        n = len(x)
        l1 = np.sum(np.abs(x))
        l2 = np.sqrt(np.sum(x**2))
        return (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)

    def nmf_analysis(self, Y, n_components=10, max_components=15, max_iter=500):
        """Fit NMF and summarize additive structure of an ensemble.

        Computes (1) reconstruction error vs number of components, (2)
        energy-based participation ratio across components, and (3) mean Hoyer
        sparsity of temporal motifs ``H``. Useful when signals are nonnegative
        and one cares about localized additive motifs rather than signed PCA.

        Parameters
        ----------
        Y : ndarray, shape (n_trials, n_timepoints)
            Nonnegative data matrix.
        n_components : int, optional
            Components for PR and sparsity. Default is 10.
        max_components : int, optional
            Max ``k`` for the reconstruction curve. Default is 15.
        max_iter : int, optional
            NMF max iterations. Default is 500.

        Returns
        -------
        dict
            ``reconstruction_curve``, ``participation_ratio``,
            ``energy_distribution``, ``sparsity``.
        """
        results = {}

        # Compute reconstruction curve (requires multiple NMF fits)
        errors = []
        for k in range(1, max_components + 1):
            model = NMF(
                n_components=k,
                init="nndsvda",
                max_iter=max_iter,
                random_state=0,
            )
            W = model.fit_transform(Y)
            H = model.components_
            Y_hat = W @ H
            error = np.linalg.norm(Y - Y_hat) / np.linalg.norm(Y)
            errors.append(error)
        results["reconstruction_curve"] = np.array(errors)

        # Compute participation ratio and sparsity using n_components
        model = NMF(
            n_components=n_components,
            init="nndsvda",
            max_iter=max_iter,
            random_state=0,
        )
        W = model.fit_transform(Y)
        H = model.components_

        # Participation ratio
        energies = []
        for k in range(n_components):
            component = np.outer(W[:, k], H[k, :])
            energy = np.linalg.norm(component, "fro") ** 2
            energies.append(energy)
        energies = np.array(energies)
        p = energies / np.sum(energies)
        results["participation_ratio"] = 1.0 / np.sum(p**2)
        results["energy_distribution"] = p

        # Sparsity
        sparsities = []
        for k in range(n_components):
            sparsities.append(self._hoyer_sparsity(H[k]))
        results["sparsity"] = np.mean(sparsities)

        return results

    def compare_nmf_analysis(self, n_components=10, max_components=15, max_iter=500):
        """Run :meth:`nmf_analysis` on A and B and return side-by-side metrics.

        Parameters
        ----------
        n_components : int, optional
            Components for PR/sparsity. Default is 10.
        max_components : int, optional
            Max components for reconstruction curves. Default is 15.
        max_iter : int, optional
            NMF iterations. Default is 500.

        Returns
        -------
        dict
            Nested dicts for ``participation_ratio``, ``reconstruction_curve``,
            and ``sparsity``, each with ``A`` / ``B`` entries.
        """
        results_A = self.nmf_analysis(self.Y_A, n_components, max_components, max_iter)
        results_B = self.nmf_analysis(self.Y_B, n_components, max_components, max_iter)

        return {
            "participation_ratio": {
                "A": results_A["participation_ratio"],
                "B": results_B["participation_ratio"],
            },
            "reconstruction_curve": {
                "A": results_A["reconstruction_curve"],
                "B": results_B["reconstruction_curve"],
            },
            "sparsity": {
                "A": results_A["sparsity"],
                "B": results_B["sparsity"],
            },
        }

    # ==========================================================
    # Legacy methods (kept for backward compatibility)
    # ==========================================================

    # def nmf_participation_ratio(self, Y, n_components=10):
    #     results = self.nmf_analysis(
    #         Y, n_components=n_components, max_components=n_components
    #     )
    #     return (results["participation_ratio"], results["energy_distribution"])

    # def compare_nmf_pr(self, n_components=10):
    #     comparison = self.compare_nmf_analysis(
    #         n_components=n_components, max_components=n_components
    #     )
    #     return comparison["participation_ratio"]

    # def nmf_reconstruction_curve(self, Y, max_components=15):
    #     results = self.nmf_analysis(
    #         Y, n_components=max_components, max_components=max_components
    #     )
    #     return results["reconstruction_curve"]

    # def compare_reconstruction(self, max_components=15):
    #     comparison = self.compare_nmf_analysis(
    #         n_components=max_components, max_components=max_components
    #     )
    #     return comparison["reconstruction_curve"]

    # def nmf_sparsity(self, Y, n_components=10):
    #     results = self.nmf_analysis(
    #         Y, n_components=n_components, max_components=n_components
    #     )
    #     return results["sparsity"]

    # def compare_sparsity(self, n_components=10):
    #     comparison = self.compare_nmf_analysis(
    #         n_components=n_components, max_components=n_components
    #     )
    #     return comparison["sparsity"]

    # ==========================================================
    # kNN Entropy (still valid)
    # ==========================================================

    def knn_entropy(self, Y, k=5):
        """kNN entropy proxy (see :meth:`BatchSignalAnalyzer.knn_entropy`).

        Parameters
        ----------
        Y : ndarray
            Trial×feature matrix.
        k : int, optional
            Neighbor index. Default is 5.

        Returns
        -------
        float
            Mean log kNN distance.
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Y)
        distances, _ = nbrs.kneighbors(Y)
        r = distances[:, -1]
        return np.mean(np.log(r + 1e-12))

    def compare_knn_entropy(self, k=5):
        """kNN entropy for conditions A and B.

        Parameters
        ----------
        k : int, optional
            Neighbor count. Default is 5.

        Returns
        -------
        dict
            ``{'A': float, 'B': float}``.
        """
        return {"A": self.knn_entropy(self.Y_A, k), "B": self.knn_entropy(self.Y_B, k)}

    # ==========================================================
    # Compression ratio
    # ==========================================================

    def compression_ratio(self, Y):
        """Mean zlib ratio after mapping trials to 8-bit (assumes ~[0, 1] scale).

        Parameters
        ----------
        Y : ndarray, shape (n_trials, n_timepoints)
            Typically already normalized trials.

        Returns
        -------
        float
            Average compressed/raw size ratio.
        """

        ratios = []

        for trial in Y:
            scaled = (trial * 255).astype(np.uint8)
            raw = scaled.tobytes()
            compressed = zlib.compress(raw)
            ratios.append(len(compressed) / len(raw))

        return np.mean(ratios)

    def compare_compression(self):
        """Compression ratios for conditions A and B.

        Returns
        -------
        dict
            ``{'A': float, 'B': float}``.
        """
        return {
            "A": self.compression_ratio(self.Y_A),
            "B": self.compression_ratio(self.Y_B),
        }

    # ==========================================================
    # Optional: PCA comparison (for sanity)
    # ==========================================================

    def pca_participation_ratio(self, Y):
        """Column-centered SVD participation ratio (signed / variance-based).

        Provided as a reference alongside NMF metrics; does not respect
        nonnegativity of the generative model.

        Parameters
        ----------
        Y : ndarray
            Trial×feature matrix.

        Returns
        -------
        float
            Participation ratio of singular values.
        """

        Yc = Y - Y.mean(axis=0)
        _, S, _ = np.linalg.svd(Yc, full_matrices=False)

        return (np.sum(S) ** 2) / np.sum(S**2)

    def compare_pca(self):
        """PCA participation ratio for conditions A and B.

        Returns
        -------
        dict
            ``{'A': float, 'B': float}``.
        """
        return {
            "A": self.pca_participation_ratio(self.Y_A),
            "B": self.pca_participation_ratio(self.Y_B),
        }

    # ==========================================================
    # Run all analyses
    # ==========================================================

    def run_all(self, n_components=10, max_iter=500):
        """Run NMF, kNN, compression, and PCA comparisons.

        Parameters
        ----------
        n_components : int, optional
            NMF rank for PR/sparsity/reconstruction. Default is 10.
        max_iter : int, optional
            NMF iterations. Default is 500.

        Returns
        -------
        dict
            ``nmf_pr``, ``nmf_rec``, ``nmf_sparsity``, ``knn_entropy``,
            ``compression``, ``pca_pr``.
        """
        nmf_results = self.compare_nmf_analysis(
            n_components=n_components, max_components=n_components, max_iter=max_iter
        )

        return {
            "nmf_pr": nmf_results["participation_ratio"],
            "nmf_rec": nmf_results["reconstruction_curve"],
            "nmf_sparsity": nmf_results["sparsity"],
            "knn_entropy": self.compare_knn_entropy(),
            "compression": self.compare_compression(),
            "pca_pr": self.compare_pca(),
        }

    def run_all_print(self, n_components=10, max_iter=500):
        """Compute :meth:`run_all` and print annotated summaries.

        Parameters
        ----------
        n_components : int, optional
            Passed to :meth:`run_all`. Default is 10.
        max_iter : int, optional
            Passed to :meth:`run_all`. Default is 500.

        Returns
        -------
        None
        """
        results = self.run_all(n_components=n_components, max_iter=max_iter)

        print("\n--- NMF Participation Ratio:")
        print("\tadditive component diversity. higher -> more distinct temporal motifs")
        print(f"A: {results['nmf_pr']['A']:.3f}")
        print(f"B: {results['nmf_pr']['B']:.3f}")
        print("\n--- NMF Reconstruction:")
        print(
            "\thow many components needed to explain signals (lower error = better fit)"
        )
        print(f"A (final error): {results['nmf_rec']['A'][-1]:.3f}")
        print(f"B (final error): {results['nmf_rec']['B'][-1]:.3f}")
        print("\n--- NMF Sparsity (Hoyer):")
        print("\tcomponent localization. higher -> more sparse / localized motifs")
        print(f"A: {results['nmf_sparsity']['A']:.3f}")
        print(f"B: {results['nmf_sparsity']['B']:.3f}")
        print("\n--- kNN Entropy:")
        print("\tdistribution complexity. higher -> more diverse signals")
        print(f"A: {results['knn_entropy']['A']:.3f}")
        print(f"B: {results['knn_entropy']['B']:.3f}")
        print("\n--- Compression Ratio:")
        print("\talgorithmic complexity. higher -> less compressible, more complex")
        print(f"A: {results['compression']['A']:.3f}")
        print(f"B: {results['compression']['B']:.3f}")
        print("\n--- PCA Participation Ratio (reference):")
        print("\tvariance-based dimensionality (not nonnegativity-aware)")
        print(f"A: {results['pca_pr']['A']:.3f}")
        print(f"B: {results['pca_pr']['B']:.3f}")
        print("\n")
