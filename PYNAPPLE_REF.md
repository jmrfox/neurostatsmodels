# Pynapple Reference Guide

This document explains how pynapple data structures are used in the `neurostatsmodels` package.

## Core Pynapple Objects

### 1. `nap.Ts` - Timestamps

**What it is**: A 1D array of timestamps (spike times for a single neuron).

**Key attributes**:

- `.t` - numpy array of timestamps
- `.time_support` - IntervalSet defining valid time ranges

**Example**:

```python
import pynapple as nap
import numpy as np

# Create timestamps
spike_times = nap.Ts(t=np.array([0.1, 0.3, 0.7, 1.2]))
print(spike_times.t)  # [0.1 0.3 0.7 1.2]
```

### 2. `nap.TsGroup` - Group of Timestamps

**What it is**: A collection of `Ts` objects, one per neuron, with optional metadata.

**Key attributes**:

- `.time_support` - IntervalSet shared across all neurons
- Access individual neurons: `tsgroup[neuron_idx]`
- Metadata via `.set_info()` and `.get_info()`

**Key methods**:

- `.count(bin_size)` - Count spikes in bins
- `.restrict(interval)` - Get spikes within time interval
- `.to_tsd(feature)` - Convert to time series

**Example**:

```python
# Create TsGroup for 3 neurons
neuron_spikes = {
    0: nap.Ts(t=np.array([0.1, 0.5])),
    1: nap.Ts(t=np.array([0.2, 0.6, 0.9])),
    2: nap.Ts(t=np.array([0.3]))
}
tsgroup = nap.TsGroup(neuron_spikes)

# Add metadata
tsgroup.set_info(mean=np.array([0, 50, 100]), sigma=np.array([10, 10, 10]))

# Access neuron 1
print(tsgroup[1].t)  # [0.2 0.6 0.9]
```

### 3. `nap.IntervalSet` - Time Intervals

**What it is**: Defines time epochs (e.g., trial boundaries).

**Key attributes**:

- `.start` - Start times of intervals
- `.end` - End times of intervals
- Access single interval: `intervalset[idx]`

**Example**:

```python
# Define 5 trials of 1 second each
trials = nap.IntervalSet(
    start=np.arange(5),      # [0, 1, 2, 3, 4]
    end=np.arange(5) + 1.0   # [1, 2, 3, 4, 5]
)

# Get trial 2
trial_2 = trials[2]
print(trial_2.start, trial_2.end)  # [2.] [3.]
```

### 4. `nap.TsdFrame` - 2D Time Series

**What it is**: Time series data with multiple columns (e.g., tuning curves for multiple neurons).

**Key attributes**:

- `.t` - Time axis (or stimulus axis in our case)
- `.values` - 2D numpy array (time × neurons)
- `.d` - Alias for `.values`

**Key methods**:

- `.interpolate(new_times)` - Interpolate to new time points
- Column access: `tsdframe[column_name]` or `tsdframe[:, col_idx]`

**Example**:

```python
# Stimulus values as "time" axis
stimulus = np.linspace(-200, 200, 100)
# Firing rates for 3 neurons
rates = np.random.rand(100, 3) * 50

tuning_curves = nap.TsdFrame(t=stimulus, d=rates)

# Interpolate at specific stimulus values
new_stim = np.array([-100, 0, 100])
interpolated = tuning_curves.interpolate(new_stim)
print(interpolated.values.shape)  # (3, 3) - 3 stimuli × 3 neurons
```

---

## How We Use Pynapple in `neurostatsmodels`

### Spike Generation (`generate_spikes`)

**Returns**: `nap.TsGroup` (single stimulus) or `dict` of `TsGroup` (multiple stimuli)

```python
from neurostatsmodels.populations import GaussianTunedPopulation
import numpy as np

pop = GaussianTunedPopulation(n_neurons=10, max_rate=50.0)
pop.set_means_uniform(np.linspace(-200, 200, 100))
pop.set_sigmas(50.0)

# Generate spikes for single stimulus
spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=5)
# spikes is a TsGroup with:
# - 10 neurons (keys 0-9)
# - time_support = 5 trial epochs
# - metadata: mean, sigma, max_rate

# Access neuron 3
neuron_3_spikes = spikes[3]
print(neuron_3_spikes.t)  # All spike times for neuron 3 across all trials

# Get spikes in trial 2
trial_2_interval = spikes.time_support[2]
neuron_3_trial_2 = spikes[3].restrict(trial_2_interval)
print(neuron_3_trial_2.t)  # Spike times for neuron 3 in trial 2 only
```

**Multiple stimuli**:

```python
# Generate spikes for 3 different stimuli
stimuli = np.array([-100, 0, 100])
spikes_dict = pop.generate_spikes(stimuli, duration=1.0, n_trials=5)
# spikes_dict = {0: TsGroup, 1: TsGroup, 2: TsGroup}

# Access stimulus 1 (s=0), neuron 3
neuron_3_stim_1 = spikes_dict[1][3]
```

### Tuning Curves (`build_tuning_curves`)

**Stores**: `nap.TsdFrame` in `self.tuning_curves`

```python
# Build tuning curves over stimulus grid
stimulus_grid = np.linspace(-200, 200, 1000)
pop.build_tuning_curves(stimulus_grid)

# Access tuning curves
# tuning_curves.t = stimulus values
# tuning_curves.values = rates (1000 stimuli × 10 neurons)

# Get rates at specific stimulus
rates_at_zero = pop.get_rates_at(0.0)  # Uses TsdFrame.interpolate()
print(rates_at_zero.shape)  # (10,) - one rate per neuron
```

### Fisher Information Curves (`build_fisher_info_curves`)

**Stores**: `nap.TsdFrame` in `self.fisher_info_curves`

```python
pop.build_fisher_info_curves()

# Query FI at specific stimulus
fi_at_zero = pop.get_fisher_info_at(0.0)  # Uses TsdFrame.interpolate()
print(fi_at_zero.shape)  # (10,) - FI for each neuron
```

### Decoding (`decode_mle`)

**Input**: `nap.TsGroup`
**Output**: numpy array of decoded values (one per trial)

```python
# Generate spikes
spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=100)

# Decode stimulus from spikes
s_hat = pop.decode_mle(spikes)
print(s_hat.shape)  # (100,) - one estimate per trial

# Analyze decoder performance
bias = np.mean(s_hat) - 0.0
std = np.std(s_hat)
print(f"Bias: {bias:.2f}, Std: {std:.2f}")
```

### Plotting (`plot_spike_raster`)

**Input**: `nap.TsGroup` and trial index
**Output**: Raster plot for single trial

```python
from neurostatsmodels.plotting import plot_spike_raster

# Generate spikes
spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=5)

# Plot trial 0
fig, ax = plot_spike_raster(spikes, trial_idx=0, title="Trial 0")

# Plot trial 3
fig, ax = plot_spike_raster(spikes, trial_idx=3, title="Trial 3")
```

---

## Key Pynapple Operations

### Restricting to Time Intervals

```python
# Get spikes only in specific trials
trial_0 = spikes.time_support[0]
trial_0_spikes = spikes[neuron_idx].restrict(trial_0)

# Restrict to custom interval
custom_interval = nap.IntervalSet(start=[0.5], end=[1.5])
restricted = spikes[neuron_idx].restrict(custom_interval)
```

### Counting Spikes

```python
# Count spikes in bins
spike_counts = spikes.count(bin_size=0.1)  # 100ms bins
# Returns TsdFrame with counts for each neuron

# Count spikes in specific interval
trial_spikes = spikes[neuron_idx].restrict(trial_interval)
n_spikes = len(trial_spikes)
```

### Working with Metadata

```python
# Set metadata on TsGroup
tsgroup.set_info(
    mean=np.array([...]),
    sigma=np.array([...]),
    preferred_stimulus=np.array([...])
)

# Get metadata
means = tsgroup.get_info('mean')
```

---

## Common Patterns

### Iterate over trials

```python
spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=10)

for trial_idx in range(len(spikes.time_support)):
    trial_interval = spikes.time_support[trial_idx]
    
    # Get spikes for all neurons in this trial
    for neuron_idx in range(pop.n_neurons):
        trial_spikes = spikes[neuron_idx].restrict(trial_interval)
        n_spikes = len(trial_spikes)
        print(f"Neuron {neuron_idx}, Trial {trial_idx}: {n_spikes} spikes")
```

### Compare across stimuli

```python
stimuli = np.array([-100, 0, 100])
spikes_dict = pop.generate_spikes(stimuli, duration=1.0, n_trials=20)

for stim_idx, stim_value in enumerate(stimuli):
    tsgroup = spikes_dict[stim_idx]
    
    # Decode this stimulus
    s_hat = pop.decode_mle(tsgroup)
    
    # Compute bias
    bias = np.mean(s_hat) - stim_value
    print(f"Stimulus {stim_value}: bias = {bias:.2f}")
```

### Extract spike counts for analysis

```python
spikes = pop.generate_spikes(0.0, duration=1.0, n_trials=50)

# Get spike counts for each neuron in each trial
spike_count_matrix = np.zeros((pop.n_neurons, 50))

for trial_idx in range(50):
    trial_interval = spikes.time_support[trial_idx]
    for neuron_idx in range(pop.n_neurons):
        trial_spikes = spikes[neuron_idx].restrict(trial_interval)
        spike_count_matrix[neuron_idx, trial_idx] = len(trial_spikes)

# Now spike_count_matrix[i, j] = count for neuron i, trial j
```

---

## Benefits of Pynapple Integration

1. **Automatic time management**: IntervalSet handles trial boundaries
2. **Metadata storage**: Store neuron properties alongside spike data
3. **Efficient operations**: `.restrict()`, `.count()` are optimized
4. **Consistent API**: All time series use same interface
5. **Analysis tools**: Built-in methods for common neuroscience operations

---

## Quick Reference

| Object            | Use Case             | Key Methods                               |
| ----------------- | -------------------- | ----------------------------------------- |
| `nap.Ts`          | Single neuron spikes | `.restrict()`, `.t`                       |
| `nap.TsGroup`     | Population spikes    | `[neuron_idx]`, `.count()`, `.set_info()` |
| `nap.IntervalSet` | Trial epochs         | `[trial_idx]`, `.start`, `.end`           |
| `nap.TsdFrame`    | Tuning curves, FI    | `.interpolate()`, `.values`               |

---

## Further Reading

- [Pynapple Documentation](https://pynapple.org)
- [Pynapple User Guide](https://pynapple.org/user_guide.html)
- [Pynapple API Reference](https://pynapple.org/api.html)
