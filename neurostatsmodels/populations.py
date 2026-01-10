# Population-related models and utilities
import numpy as np
import pynapple as nap


class GaussianTunedPopulation:
    """Population of neurons with Gaussian tuning curves.

    Each neuron has a tuning curve r(s) = max_rate * exp(-0.5 * ((s - mean) / sigma)^2)
    where s is the stimulus value, and max_rate = reference_rate * (reference_sigma / sigma).

    Parameters
    ----------
    n_neurons : int
        Number of neurons in the population
    reference_rate : float
        Reference firing rate in Hz. Used to compute max_rate for each neuron.
    reference_sigma : float
        Reference tuning width. Used to compute max_rate for each neuron.
    """

    def __init__(self, n_neurons, reference_rate=100.0, reference_sigma=50.0):
        self.n_neurons = n_neurons
        self.reference_rate = reference_rate
        self.reference_sigma = reference_sigma

        self.means = None
        self.sigmas = None
        self.refractory_periods = None
        self.spontaneous_rates = None

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

    def set_refractory_periods(self, refractory_periods):
        """Set the refractory periods for each neuron.

        Parameters
        ----------
        refractory_periods : float or array-like
            Refractory periods in seconds. If float, all neurons have the same
            refractory period. If array-like, must have length n_neurons.
        """
        if np.isscalar(refractory_periods):
            self.refractory_periods = np.full(
                self.n_neurons, refractory_periods, dtype=float
            )
        else:
            self.refractory_periods = np.asarray(refractory_periods, dtype=float)
            if len(self.refractory_periods) != self.n_neurons:
                raise ValueError(
                    f"refractory_periods must have length {self.n_neurons}"
                )

    def set_spontaneous_rates(self, spontaneous_rates):
        """Set the spontaneous (baseline) firing rates for each neuron.

        Parameters
        ----------
        spontaneous_rates : float or array-like
            Spontaneous firing rates in Hz. If float, all neurons have the same
            spontaneous rate. If array-like, must have length n_neurons.
        """
        if np.isscalar(spontaneous_rates):
            self.spontaneous_rates = np.full(
                self.n_neurons, spontaneous_rates, dtype=float
            )
        else:
            self.spontaneous_rates = np.asarray(spontaneous_rates, dtype=float)
            if len(self.spontaneous_rates) != self.n_neurons:
                raise ValueError(f"spontaneous_rates must have length {self.n_neurons}")

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

        # Compute max_rate for each neuron based on its sigma
        max_rate = self.reference_rate * (self.reference_sigma / self.sigmas)

        # Broadcast: (n_neurons, 1) - (1, n_stimuli) -> (n_neurons, n_stimuli)
        z = (stimulus[np.newaxis, :] - self.means[:, np.newaxis]) / self.sigmas[
            :, np.newaxis
        ]
        rates = max_rate[:, np.newaxis] * np.exp(-0.5 * z**2)

        # Add spontaneous rate if set
        if self.spontaneous_rates is not None:
            rates = rates + self.spontaneous_rates[:, np.newaxis]

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

        # Interpolate each neuron's tuning curve at the stimulus values
        rates = np.zeros((self.n_neurons, len(stimulus)))
        for i in range(self.n_neurons):
            rates[i, :] = np.interp(
                stimulus, self.stimulus_grid, self.tuning_curves.values[:, i]
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

        # Interpolate each neuron's FI curve at the stimulus values
        fisher_info = np.zeros((self.n_neurons, len(stimulus)))
        for i in range(self.n_neurons):
            fisher_info[i, :] = np.interp(
                stimulus, self.stimulus_grid, self.fisher_info_curves.values[:, i]
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

    def generate_spikes(
        self,
        stimulus,
        duration=1.0,
        n_trials=1,
        dt=0.001,
        alpha=0.0,
        tau=0.1,
        random_state=None,
    ):
        """Generate spike times using time-stepping with dynamic modulation.

        Spikes are generated by stepping through time in increments of dt and computing
        the probability of a spike at each time step. The spike probability for neuron i is:
        p_i(t) = g(t) * r_{i,base}(s) * f_i(t - t_{i,last}) * dt

        where:
        - g(t) = 1 / (1 + alpha * A(t)) is global gain modulation
        - r_{i,base}(s) is the tuning curve (base firing rate)
        - f_i(t - t_{i,last}) is the refractory kernel (0 during refractory, 1 after)
        - A(t) is the activity trace: A_{t+1} = A_t * exp(-dt/tau) + sum_i n_i(t)

        Parameters
        ----------
        stimulus : float or array-like
            Stimulus value(s) to generate spikes for.
        duration : float, optional
            Duration of each trial in seconds. Default is 1.0.
        n_trials : int, optional
            Number of trials to generate. Default is 1.
        dt : float, optional
            Time step size in seconds. Default is 0.001 (1 ms).
        alpha : float, optional
            Suppression parameter for global gain modulation. Default is 0.0 (no suppression).
        tau : float, optional
            Time constant for activity trace relaxation in seconds. Default is 0.1.
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
        >>> # Single stimulus with default dynamics
        >>> spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=5)
        >>> # Access neuron 3
        >>> spike_train = spikes[3]
        >>>
        >>> # With global gain suppression
        >>> spikes = pop.generate_spikes(0.0, duration=1.0, alpha=0.01, tau=0.05)
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

        # Validate parameters
        if duration <= 0:
            raise ValueError("duration must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if dt > duration:
            raise ValueError("dt must be smaller than duration")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if tau <= 0:
            raise ValueError("tau must be positive")

        # Compute base firing rates from tuning curves
        rates = self.compute_rates(stimulus)
        is_scalar = rates.ndim == 1

        # Create trial epochs as IntervalSet
        gap = 1e-6  # 1 microsecond gap between trials
        starts = np.arange(n_trials, dtype=np.float64) * (duration + gap)
        ends = starts + duration
        trial_epochs = nap.IntervalSet(start=starts, end=ends)

        # Number of time steps per trial
        n_steps = int(np.ceil(duration / dt))

        # Precompute decay factor for activity trace
        decay_factor = np.exp(-dt / tau)

        if is_scalar:
            # Single stimulus: generate spikes for all neurons across all trials
            # Initialize spike time storage for each neuron
            neuron_spikes = {}
            for neuron_idx in range(self.n_neurons):
                neuron_spikes[neuron_idx] = []

            # Get refractory periods for all neurons
            refracs = np.zeros(self.n_neurons)
            if self.refractory_periods is not None:
                refracs = self.refractory_periods

            for trial_idx in range(n_trials):
                trial_start = starts[trial_idx]

                # Initialize activity trace (scalar for whole population) and last spike times per neuron
                A = 0.0
                t_last_spikes = np.full(self.n_neurons, -np.inf)

                # Step through time
                for step in range(n_steps):
                    t = step * dt

                    # Compute global gain modulation (same for all neurons)
                    g = 1.0 / (1.0 + alpha * A)

                    # Track spikes at this time step for activity trace update
                    n_spikes_this_step = 0

                    # Process each neuron
                    for neuron_idx in range(self.n_neurons):
                        base_rate = rates[neuron_idx]

                        # Compute refractory kernel: 0 during refractory, 1 after
                        time_since_last = t - t_last_spikes[neuron_idx]
                        f_refrac = 0.0 if time_since_last < refracs[neuron_idx] else 1.0

                        # Compute spike probability
                        spike_prob = g * base_rate * f_refrac * dt

                        # Clip probability to [0, 1] for numerical stability
                        spike_prob = np.clip(spike_prob, 0.0, 1.0)

                        # Sample spike occurrence
                        if rng.random() < spike_prob:
                            neuron_spikes[neuron_idx].append(trial_start + t)
                            t_last_spikes[neuron_idx] = t
                            n_spikes_this_step += 1

                    # Update activity trace based on total spikes at this time step
                    A = A * decay_factor + n_spikes_this_step

            # Convert spike lists to Ts objects
            for neuron_idx in range(self.n_neurons):
                neuron_spikes[neuron_idx] = nap.Ts(
                    t=np.array(neuron_spikes[neuron_idx]), time_support=trial_epochs
                )

            # Create TsGroup with metadata
            tsgroup = nap.TsGroup(neuron_spikes, time_support=trial_epochs)
            metadata = {
                "mean": self.means,
                "sigma": self.sigmas,
                "reference_rate": np.full(self.n_neurons, self.reference_rate),
                "reference_sigma": np.full(self.n_neurons, self.reference_sigma),
            }
            if self.refractory_periods is not None:
                metadata["refractory_period"] = self.refractory_periods
            if self.spontaneous_rates is not None:
                metadata["spontaneous_rate"] = self.spontaneous_rates
            tsgroup.set_info(**metadata)
            return tsgroup

        else:
            # Multiple stimuli: return dict of TsGroups
            n_stimuli = rates.shape[1]
            result = {}

            for stim_idx in range(n_stimuli):
                # Initialize spike time storage for each neuron
                neuron_spikes = {}
                for neuron_idx in range(self.n_neurons):
                    neuron_spikes[neuron_idx] = []

                # Get refractory periods for all neurons
                refracs = np.zeros(self.n_neurons)
                if self.refractory_periods is not None:
                    refracs = self.refractory_periods

                for trial_idx in range(n_trials):
                    trial_start = starts[trial_idx]

                    # Initialize activity trace (scalar for whole population) and last spike times per neuron
                    A = 0.0
                    t_last_spikes = np.full(self.n_neurons, -np.inf)

                    # Step through time
                    for step in range(n_steps):
                        t = step * dt

                        # Compute global gain modulation (same for all neurons)
                        g = 1.0 / (1.0 + alpha * A)

                        # Track spikes at this time step for activity trace update
                        n_spikes_this_step = 0

                        # Process each neuron
                        for neuron_idx in range(self.n_neurons):
                            base_rate = rates[neuron_idx, stim_idx]

                            # Compute refractory kernel: 0 during refractory, 1 after
                            time_since_last = t - t_last_spikes[neuron_idx]
                            f_refrac = (
                                0.0 if time_since_last < refracs[neuron_idx] else 1.0
                            )

                            # Compute spike probability
                            spike_prob = g * base_rate * f_refrac * dt

                            # Clip probability to [0, 1] for numerical stability
                            spike_prob = np.clip(spike_prob, 0.0, 1.0)

                            # Sample spike occurrence
                            if rng.random() < spike_prob:
                                neuron_spikes[neuron_idx].append(trial_start + t)
                                t_last_spikes[neuron_idx] = t
                                n_spikes_this_step += 1

                        # Update activity trace based on total spikes at this time step
                        A = A * decay_factor + n_spikes_this_step

                # Convert spike lists to Ts objects
                for neuron_idx in range(self.n_neurons):
                    neuron_spikes[neuron_idx] = nap.Ts(
                        t=np.array(neuron_spikes[neuron_idx]), time_support=trial_epochs
                    )

                tsgroup = nap.TsGroup(neuron_spikes, time_support=trial_epochs)
                metadata = {
                    "mean": self.means,
                    "sigma": self.sigmas,
                    "reference_rate": np.full(self.n_neurons, self.reference_rate),
                    "reference_sigma": np.full(self.n_neurons, self.reference_sigma),
                }
                if self.refractory_periods is not None:
                    metadata["refractory_period"] = self.refractory_periods
                if self.spontaneous_rates is not None:
                    metadata["spontaneous_rate"] = self.spontaneous_rates
                tsgroup.set_info(**metadata)
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
