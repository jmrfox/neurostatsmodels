import numpy as np
import matplotlib.pyplot as plt


def plot_spike_raster(
    tsgroup,
    trial_idx=None,
    neuron_labels=None,
    figsize=None,
    title=None,
    xlabel="Time (s)",
    ylabel="Neuron",
    ax=None,
):
    """Plot spike raster for a single trial of multi-neuron population response.

    Parameters
    ----------
    tsgroup : nap.TsGroup
        TsGroup containing spike times for all neurons. Trial epochs are
        extracted from the time_support attribute.
    trial_idx : int or None, optional
        Which trial to plot. If None and there is only one trial, plots that trial.
        If None and there are multiple trials, defaults to 0. Default is None.
    neuron_labels : list of str, optional
        Labels for each neuron. If None, uses indices 0, 1, 2, ...
    figsize : tuple, optional
        Figure size (width, height). If None, auto-computed based on data.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label. Default is 'Time (s)'.
    ylabel : str, optional
        Y-axis label. Default is 'Neuron'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    Examples
    --------
    >>> # Create population and generate spikes
    >>> pop = GaussianTunedPopulation(n_neurons=10, max_rate=50.0)
    >>> pop.set_means_uniform(np.linspace(-200, 200, 100))
    >>> pop.set_sigmas(50.0)
    >>>
    >>> # Generate spikes with multiple trials
    >>> spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=5)
    >>> fig, ax = plot_spike_raster(spikes, trial_idx=0)
    >>>
    >>> # Generate spikes with single trial
    >>> spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=1)
    >>> fig, ax = plot_spike_raster(spikes)  # trial_idx not needed
    """
    # Get trial interval
    trial_epochs = tsgroup.time_support
    n_trials = len(trial_epochs)

    # Handle trial_idx
    if trial_idx is None:
        trial_idx = 0

    if trial_idx >= n_trials:
        raise ValueError(
            f"trial_idx={trial_idx} but only {n_trials} trial(s) available"
        )

    trial_interval = trial_epochs[trial_idx]
    trial_start = trial_interval.start[0]
    trial_end = trial_interval.end[0]
    trial_duration = trial_end - trial_start

    n_neurons = len(tsgroup)
    neuron_indices = sorted(tsgroup.keys())

    # Create figure if needed
    if ax is None:
        if figsize is None:
            width = 10
            height = max(4, n_neurons * 0.3)
            figsize = (width, height)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Generate neuron labels
    if neuron_labels is None:
        neuron_labels = [str(i) for i in neuron_indices]

    # Collect spike times and positions for this trial
    for plot_idx, neuron_idx in enumerate(neuron_indices):
        # Restrict spikes to this trial
        trial_spikes = tsgroup[neuron_idx].restrict(trial_interval)

        if len(trial_spikes) > 0:
            # Convert to relative time within trial
            times = trial_spikes.t - trial_start

            # Plot spikes as vertical lines
            for time in times:
                ax.plot(
                    [time, time], [plot_idx - 0.4, plot_idx + 0.4], "k-", linewidth=1
                )

    # Set axis properties
    ax.set_xlim(0, trial_duration)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.set_yticks(range(n_neurons))
    ax.set_yticklabels(neuron_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(False)
    ax.invert_yaxis()

    return fig, ax
