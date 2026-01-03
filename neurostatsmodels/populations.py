# Population-related models and utilities
import numpy as np


def gaussian(variable, mean=0, std=1):
    """Compute Gaussian probability density function."""
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((variable - mean) / std) ** 2
    )


class GaussianTunedPopulation:
    """Population of neurons with Gaussian tuning curves.

    Each neuron has a tuning curve r(s) = max_rate * exp(-0.5 * ((s - mean) / sigma)^2)
    where s is the stimulus value.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population
    max_rate : float or array-like
        Maximum firing rate(s) in Hz. If float, all neurons have the same max_rate.
        If array-like, must have length n_neurons.
    """

    def __init__(self, n_neurons, max_rate=100.0):
        self.n_neurons = n_neurons

        if np.isscalar(max_rate):
            self.max_rate = np.full(n_neurons, max_rate, dtype=float)
        else:
            self.max_rate = np.asarray(max_rate, dtype=float)
            if len(self.max_rate) != n_neurons:
                raise ValueError(f"max_rate must have length {n_neurons}")

        self.means = None
        self.sigmas = None

        self.stimulus_grid = None
        self.tuning_curves = None
        self.fisher_info_curves = None

    def set_means(self, means):
        """Set the preferred stimulus values (means) for each neuron.

        Parameters
        ----------
        means : array-like
            Preferred stimulus values. Must have length n_neurons.
        """
        self.means = np.asarray(means, dtype=float)
        if len(self.means) != self.n_neurons:
            raise ValueError(f"means must have length {self.n_neurons}")

    def set_means_uniform(self, stimulus_values):
        """Set means uniformly distributed over given stimulus values.

        Parameters
        ----------
        stimulus_values : array-like
            Grid of stimulus values to distribute means over.
        """
        stimulus_values = np.asarray(stimulus_values)
        if len(stimulus_values) < self.n_neurons:
            raise ValueError(f"Need at least {self.n_neurons} stimulus values")

        indices = np.linspace(0, len(stimulus_values) - 1, self.n_neurons, dtype=int)
        self.means = stimulus_values[indices]

    def set_sigmas(self, sigmas):
        """Set the tuning curve widths (standard deviations) for each neuron.

        Parameters
        ----------
        sigmas : float or array-like
            Standard deviations. If float, all neurons have the same sigma.
            If array-like, must have length n_neurons.
        """
        if np.isscalar(sigmas):
            self.sigmas = np.full(self.n_neurons, sigmas, dtype=float)
        else:
            self.sigmas = np.asarray(sigmas, dtype=float)
            if len(self.sigmas) != self.n_neurons:
                raise ValueError(f"sigmas must have length {self.n_neurons}")

    def compute_rates(self, stimulus):
        """Compute firing rates for all neurons at given stimulus value(s).

        Parameters
        ----------
        stimulus : float or array-like
            Stimulus value(s) to compute rates for.

        Returns
        -------
        rates : ndarray
            If stimulus is scalar: shape (n_neurons,)
            If stimulus is array: shape (n_neurons, n_stimuli)
        """
        if self.means is None or self.sigmas is None:
            raise ValueError("Must set means and sigmas before computing rates")

        stimulus = np.asarray(stimulus)
        is_scalar = stimulus.ndim == 0

        if is_scalar:
            stimulus = stimulus.reshape(1)

        # Broadcast: (n_neurons, 1) - (1, n_stimuli) -> (n_neurons, n_stimuli)
        z = (stimulus[np.newaxis, :] - self.means[:, np.newaxis]) / self.sigmas[
            :, np.newaxis
        ]
        rates = self.max_rate[:, np.newaxis] * np.exp(-0.5 * z**2)

        if is_scalar:
            return rates.squeeze()
        return rates

    def build_tuning_curves(self, stimulus_grid):
        """Precompute and store tuning curves over a stimulus grid.

        This method computes the full tuning curves for all neurons and stores them
        for efficient querying and derivative computation.

        Parameters
        ----------
        stimulus_grid : array-like
            Grid of stimulus values to compute tuning curves over.
        """
        self.stimulus_grid = np.asarray(stimulus_grid)
        self.tuning_curves = self.compute_rates(self.stimulus_grid)

    def build_fisher_info_curves(self, epsilon=1e-10):
        """Precompute and store Fisher information curves.

        This method uses the stored tuning curves to compute Fisher information
        curves using numerical differentiation. Must call build_tuning_curves first.

        Parameters
        ----------
        epsilon : float, optional
            Small constant to avoid division by zero. Default is 1e-10.
        """
        if self.tuning_curves is None or self.stimulus_grid is None:
            raise ValueError("Must call build_tuning_curves before building FI curves")

        # Compute derivative for each neuron's tuning curve
        dr_ds = np.gradient(self.tuning_curves, self.stimulus_grid, axis=1)

        # Compute Fisher information
        self.fisher_info_curves = dr_ds**2 / (self.tuning_curves + epsilon)

    def get_rates_at(self, stimulus):
        """Get firing rates at specific stimulus value(s) using stored tuning curves.

        Parameters
        ----------
        stimulus : float or array-like
            Stimulus value(s) to query.

        Returns
        -------
        rates : ndarray
            If stimulus is scalar: shape (n_neurons,)
            If stimulus is array: shape (n_neurons, n_stimuli)
        """
        if self.tuning_curves is None or self.stimulus_grid is None:
            raise ValueError("Must call build_tuning_curves first")

        stimulus = np.asarray(stimulus)
        is_scalar = stimulus.ndim == 0

        if is_scalar:
            stimulus = np.array([stimulus])

        # Interpolate for each neuron
        rates = np.zeros((self.n_neurons, len(stimulus)))
        for i in range(self.n_neurons):
            rates[i, :] = np.interp(
                stimulus, self.stimulus_grid, self.tuning_curves[i, :]
            )

        if is_scalar:
            return rates.squeeze()
        return rates

    def get_fisher_info_at(self, stimulus):
        """Get Fisher information at specific stimulus value(s) using stored FI curves.

        Parameters
        ----------
        stimulus : float or array-like
            Stimulus value(s) to query.

        Returns
        -------
        fisher_info : ndarray
            If stimulus is scalar: shape (n_neurons,)
            If stimulus is array: shape (n_neurons, n_stimuli)
        """
        if self.fisher_info_curves is None or self.stimulus_grid is None:
            raise ValueError("Must call build_fisher_info_curves first")

        stimulus = np.asarray(stimulus)
        is_scalar = stimulus.ndim == 0

        if is_scalar:
            stimulus = np.array([stimulus])

        # Interpolate for each neuron
        fisher_info = np.zeros((self.n_neurons, len(stimulus)))
        for i in range(self.n_neurons):
            fisher_info[i, :] = np.interp(
                stimulus, self.stimulus_grid, self.fisher_info_curves[i, :]
            )

        if is_scalar:
            return fisher_info.squeeze()
        return fisher_info

    def get_population_fisher_info_at(self, stimulus):
        """Get total population Fisher information at specific stimulus value(s).

        Parameters
        ----------
        stimulus : float or array-like
            Stimulus value(s) to query.

        Returns
        -------
        population_fi : float or ndarray
            If stimulus is scalar: float
            If stimulus is array: shape (n_stimuli,)
        """
        fisher_info = self.get_fisher_info_at(stimulus)

        # Sum over neurons (axis 0)
        if fisher_info.ndim == 1:
            return np.sum(fisher_info)
        return np.sum(fisher_info, axis=0)

    def compute_fisher_information(self, stimulus, epsilon=1e-10):
        """Compute Fisher information for each neuron at given stimulus value(s).

        Fisher information: FI(s) = (dr/ds)^2 / (r(s) + epsilon)

        Parameters
        ----------
        stimulus : float or array-like
            Stimulus value(s) to compute FI for.
        epsilon : float, optional
            Small constant to avoid division by zero. Default is 1e-10.

        Returns
        -------
        fisher_info : ndarray
            If stimulus is scalar: shape (n_neurons,)
            If stimulus is array: shape (n_neurons, n_stimuli)
        """
        stimulus = np.asarray(stimulus)
        is_scalar = stimulus.ndim == 0

        if is_scalar:
            stimulus = stimulus.reshape(1)

        rates = self.compute_rates(stimulus)

        # Compute derivative using numpy gradient
        # gradient operates on last axis by default
        dr_ds = np.gradient(rates, stimulus, axis=1)

        fisher_info = dr_ds**2 / (rates + epsilon)

        if is_scalar:
            return fisher_info.squeeze()
        return fisher_info

    def compute_population_fisher_information(self, stimulus, epsilon=1e-10):
        """Compute total Fisher information across population at given stimulus value(s).

        Total FI is the sum of individual neuron FI values.

        Parameters
        ----------
        stimulus : float or array-like
            Stimulus value(s) to compute population FI for.
        epsilon : float, optional
            Small constant to avoid division by zero. Default is 1e-10.

        Returns
        -------
        population_fi : float or ndarray
            If stimulus is scalar: float
            If stimulus is array: shape (n_stimuli,)
        """
        fisher_info = self.compute_fisher_information(stimulus, epsilon=epsilon)

        # Sum over neurons (axis 0)
        if fisher_info.ndim == 1:
            return np.sum(fisher_info)
        return np.sum(fisher_info, axis=0)
