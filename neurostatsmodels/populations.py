# Population-related models and utilities
import numpy as np
import pynapple as nap


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
        """Precompute and store tuning curves over a stimulus grid as TsdFrame.

        This method computes the full tuning curves for all neurons and stores them
        as a pynapple TsdFrame for efficient querying and derivative computation.

        Parameters
        ----------
        stimulus_grid : array-like
            Grid of stimulus values to compute tuning curves over.
        """
        self.stimulus_grid = np.asarray(stimulus_grid)
        rates = self.compute_rates(self.stimulus_grid)  # shape: (n_neurons, n_stim)
        # Store as TsdFrame: time axis = stimulus values, columns = neurons
        self.tuning_curves = nap.TsdFrame(t=self.stimulus_grid, d=rates.T)

    def build_fisher_info_curves(self, epsilon=1e-10):
        """Precompute and store Fisher information curves as TsdFrame.

        This method uses the stored tuning curves to compute Fisher information
        curves using numerical differentiation. Must call build_tuning_curves first.

        Parameters
        ----------
        epsilon : float, optional
            Small constant to avoid division by zero. Default is 1e-10.
        """
        if self.tuning_curves is None or self.stimulus_grid is None:
            raise ValueError("Must call build_tuning_curves before building FI curves")

        # Extract rates from TsdFrame
        rates = self.tuning_curves.values.T  # shape: (n_neurons, n_stim)
        
        # Compute derivative for each neuron's tuning curve
        dr_ds = np.gradient(rates, self.stimulus_grid, axis=1)

        # Compute Fisher information
        fi = dr_ds**2 / (rates + epsilon)
        
        # Store as TsdFrame
        self.fisher_info_curves = nap.TsdFrame(t=self.stimulus_grid, d=fi.T)

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

        # Wrap stimulus in Tsd for interpolation
        stimulus_tsd = nap.Tsd(t=stimulus, d=np.zeros(len(stimulus)))
        interpolated = self.tuning_curves.interpolate(stimulus_tsd, ep=None)
        rates = interpolated.values.T  # shape: (n_neurons, n_stimuli)

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

        # Wrap stimulus in Tsd for interpolation
        stimulus_tsd = nap.Tsd(t=stimulus, d=np.zeros(len(stimulus)))
        interpolated = self.fisher_info_curves.interpolate(stimulus_tsd, ep=None)
        fisher_info = interpolated.values.T  # shape: (n_neurons, n_stimuli)

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

    def generate_spikes(self, stimulus, duration=1.0, n_trials=1, random_state=None):
        """Generate spike times from a Poisson point process for the population.

        Uses exponential inter-spike intervals to generate spike times for each neuron
        based on their firing rates at the given stimulus value(s).

        Parameters
        ----------
        stimulus : float or array-like
            Stimulus value(s) to generate spikes for.
        duration : float, optional
            Duration of each trial in seconds. Default is 1.0.
        n_trials : int, optional
            Number of trials to generate. Default is 1.
        random_state : int or np.random.Generator, optional
            Random state for reproducibility.

        Returns
        -------
        spikes : nap.TsGroup or dict of nap.TsGroup
            - If stimulus is scalar: Single TsGroup with one Ts per neuron
            - If stimulus is array: dict {stimulus_idx: TsGroup} for each stimulus
            
            Each TsGroup contains spike times for all neurons across all trials.
            Trial epochs are stored in the time_support attribute.

        Examples
        --------
        >>> # Single stimulus
        >>> spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=5)
        >>> # Access neuron 3
        >>> spike_train = spikes[3]
        >>> 
        >>> # Multiple stimuli
        >>> spikes = pop.generate_spikes([0, 50, 100], duration=1.0, n_trials=5)
        >>> # Access stimulus 1, neuron 3
        >>> spike_train = spikes[1][3]
        """
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state

        # Compute firing rates
        rates = self.compute_rates(stimulus)
        is_scalar = rates.ndim == 1
        
        # Create trial epochs as IntervalSet
        trial_epochs = nap.IntervalSet(
            start=np.arange(n_trials) * duration,
            end=(np.arange(n_trials) + 1) * duration
        )

        if is_scalar:
            # Single stimulus: generate spikes for all neurons across all trials
            neuron_spikes = {}
            for neuron_idx in range(self.n_neurons):
                rate = rates[neuron_idx]
                all_times = []
                
                for trial_idx in range(n_trials):
                    trial_start = trial_idx * duration
                    if rate > 0:
                        t = 0.0
                        while t < duration:
                            isi = rng.exponential(1.0 / rate)
                            t += isi
                            if t < duration:
                                all_times.append(trial_start + t)
                
                neuron_spikes[neuron_idx] = nap.Ts(t=np.array(all_times), time_support=trial_epochs)
            
            # Create TsGroup with metadata
            tsgroup = nap.TsGroup(neuron_spikes, time_support=trial_epochs)
            tsgroup.set_info(mean=self.means, sigma=self.sigmas, max_rate=self.max_rate)
            return tsgroup
            
        else:
            # Multiple stimuli: return dict of TsGroups
            n_stimuli = rates.shape[1]
            result = {}
            
            for stim_idx in range(n_stimuli):
                neuron_spikes = {}
                for neuron_idx in range(self.n_neurons):
                    rate = rates[neuron_idx, stim_idx]
                    all_times = []
                    
                    for trial_idx in range(n_trials):
                        trial_start = trial_idx * duration
                        if rate > 0:
                            t = 0.0
                            while t < duration:
                                isi = rng.exponential(1.0 / rate)
                                t += isi
                                if t < duration:
                                    all_times.append(trial_start + t)
                    
                    neuron_spikes[neuron_idx] = nap.Ts(t=np.array(all_times), time_support=trial_epochs)
                
                tsgroup = nap.TsGroup(neuron_spikes, time_support=trial_epochs)
                tsgroup.set_info(mean=self.means, sigma=self.sigmas, max_rate=self.max_rate)
                result[stim_idx] = tsgroup
            
            return result

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

    def decode_mle(self, tsgroup, stimulus_range=None):
        """Decode stimulus using maximum likelihood estimation for each trial.

        For Poisson neurons, the log-likelihood is:
        log L(s) = sum_i [n_i * log(r_i(s)) - r_i(s) * T]
        where n_i is spike count for neuron i, r_i(s) is firing rate, T is duration.

        Parameters
        ----------
        tsgroup : nap.TsGroup
            TsGroup containing spike times for all neurons. Trial epochs are
            extracted from the time_support attribute.
        stimulus_range : tuple, optional
            (min_stimulus, max_stimulus) to search over. If None, uses the range
            of the stored tuning curve grid.

        Returns
        -------
        s_mle : ndarray
            Maximum likelihood estimates, one per trial. Shape: (n_trials,)
        """
        if self.means is None or self.sigmas is None:
            raise ValueError("Must set means and sigmas before decoding")

        # Extract trial epochs from time_support
        trial_epochs = tsgroup.time_support
        n_trials = len(trial_epochs)
        
        # Define stimulus grid to search over
        if stimulus_range is None:
            if self.stimulus_grid is not None:
                stim_grid = self.stimulus_grid
            else:
                # Default: use range around neuron means
                stim_min = np.min(self.means) - 3 * np.max(self.sigmas)
                stim_max = np.max(self.means) + 3 * np.max(self.sigmas)
                stim_grid = np.linspace(stim_min, stim_max, 1000)
        else:
            stim_grid = np.linspace(stimulus_range[0], stimulus_range[1], 1000)

        # Compute firing rates for all stimulus values
        rates = self.compute_rates(stim_grid)  # shape: (n_neurons, n_stim)
        
        # Decode each trial independently
        s_mle = np.zeros(n_trials)
        
        for trial_idx in range(n_trials):
            # Get trial interval
            trial_start = trial_epochs.start[trial_idx]
            trial_end = trial_epochs.end[trial_idx]
            duration = trial_end - trial_start
            
            # Count spikes for this trial using restrict
            spike_counts = np.zeros(self.n_neurons)
            for neuron_idx in range(self.n_neurons):
                trial_spikes = tsgroup[neuron_idx].restrict(trial_epochs[trial_idx])
                spike_counts[neuron_idx] = len(trial_spikes)
            
            # Compute log-likelihood for each stimulus value
            log_likelihood = np.zeros(len(stim_grid))
            for i in range(len(stim_grid)):
                rate_vec = rates[:, i]
                # Add small epsilon to avoid log(0)
                log_likelihood[i] = np.sum(
                    spike_counts * np.log(rate_vec + 1e-10) - rate_vec * duration
                )
            
            # Find stimulus with maximum log-likelihood for this trial
            s_mle[trial_idx] = stim_grid[np.argmax(log_likelihood)]
        
        return s_mle
