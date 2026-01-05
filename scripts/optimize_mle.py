import numpy as np
from scipy.optimize import minimize
from neurostatsmodels.populations import GaussianTunedPopulation

# Fixed parameters
n_neurons = 5
reference_rate = 50.0  # Hz, same for all neurons
reference_sigma = 30.0  # Hz, same for all neurons
spontaneous_rate = 3.0  # Hz, baseline firing rate
refractory_period = 0.005  # 5 ms refractory period
itd_range = (-200, 200)  # microseconds
n_itd_steps = 401
rate_budget = 300.0  # Total average firing rate across population (Hz)

# ITD range in microseconds
itd_min, itd_max = itd_range
itd_grid = np.linspace(itd_min, itd_max, n_itd_steps)

# Initialize population
pop = GaussianTunedPopulation(
    n_neurons=n_neurons, reference_rate=reference_rate, reference_sigma=reference_sigma
)

# Set fixed parameters
pop.set_spontaneous_rates(spontaneous_rate)
pop.set_refractory_periods(refractory_period)

# Distribute neuron means uniformly across ITD range
pop.set_means_uniform(itd_grid)

print(f"Population setup:")
print(f"  Neurons: {n_neurons}")
print(f"  Reference rate: {reference_rate} Hz")
print(f"  Reference sigma: {reference_sigma} Hz")
print(f"  Spontaneous rate: {spontaneous_rate} Hz")
print(f"  Refractory period: {refractory_period*1000} ms")
print(f"  ITD range: {itd_min} to {itd_max} μs")
print(f"  Rate budget: {rate_budget} Hz")
print(f"  Neuron preferred ITDs: {pop.means}")
print()


def objective_and_constraint(
    sigma, n_test_stimuli=10, n_trials_per_stim=10, trial_duration=1.0, random_seed=42
):
    """
    Compute MSE from MLE decoding and rate constraint violation.

    Parameters
    ----------
    sigma : float
        Tuning curve width shared for all neurons.
    n_test_stimuli : int
        Number of test ITD values to sample.
    n_trials_per_stim : int
        Number of trials per stimulus for decoding.
    trial_duration : float
        Duration of each trial in seconds.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    mse : float
        Mean squared error of MLE decoder predictions.
    rate_violation : float
        Constraint violation (should be <= 0).
    """
    # Set sigmas
    if np.isscalar(sigma):
        pop.set_sigmas(sigma)
    else:
        pop.set_sigmas(sigma[0])

    # Build tuning curves for decoding
    pop.build_tuning_curves(itd_grid)

    # Sample test ITD values uniformly from range
    rng = np.random.default_rng(random_seed)
    test_itds = rng.uniform(itd_min, itd_max, n_test_stimuli)

    # Generate spikes and decode for each test ITD
    squared_errors = []

    for true_itd in test_itds:
        # Generate spikes for this ITD
        spikes = pop.generate_spikes(
            stimulus=true_itd,
            duration=trial_duration,
            n_trials=n_trials_per_stim,
            random_state=rng,
        )

        # Decode using MLE
        decoded_itds = pop.decode_mle(spikes, stimulus_range=itd_range)

        # Compute squared errors for all trials
        squared_errors.extend((decoded_itds - true_itd) ** 2)

    # Compute MSE
    mse = np.mean(squared_errors)

    # Compute average firing rate across population and ITD range
    rates = pop.compute_rates(itd_grid)  # shape: (n_neurons, n_itd)
    avg_rate_per_neuron = np.mean(rates, axis=1)  # average over ITD
    total_avg_rate = np.sum(avg_rate_per_neuron)  # sum over neurons

    # if total_avg_rate > rate_budget, then constraint is negative (violation)
    # if total_avg_rate < rate_budget, then constraint is positive (not violated)
    constraint = rate_budget - total_avg_rate

    # Return MSE and constraint violation
    return mse, constraint


def objective(sigma):
    """Objective function: MSE from MLE decoding."""
    mse, _ = objective_and_constraint(sigma)
    return mse


def constraint_func(sigma):
    """Constraint function: rate budget - total rate (should be >= 0)."""
    _, constraint = objective_and_constraint(sigma)
    return constraint  # scipy expects constraint >= 0 (not violated)


# Initial guess:
sigma_init = 100.0

# Bounds: sigmas must be positive and reasonable
sigma_bounds = [(1.0, 200.0)]

# Constraint: total average firing rate <= rate_budget
constraints = {"type": "ineq", "fun": constraint_func}

print("Starting optimization...")
print(f"Initial sigma: {sigma_init}")

# Evaluate initial objective
print("Evaluating initial MSE (this may take a moment)...")
initial_mse, _ = objective_and_constraint(sigma_init)
pop.set_sigmas(sigma_init)
initial_rate = np.sum(np.mean(pop.compute_rates(itd_grid), axis=1))

print(f"Initial MSE: {initial_mse:.2f} μs²")
print(f"Initial RMSE: {np.sqrt(initial_mse):.2f} μs")
print(f"Initial total rate: {initial_rate:.2f} Hz")
print()

# Optimize
result = minimize(
    objective,
    sigma_init,
    method="SLSQP",
    bounds=sigma_bounds,
    constraints=constraints,
    options={"maxiter": 200, "ftol": 1e-9, "disp": True},
)

print("Optimization complete!")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Iterations: {result.nit}")
print()

# Extract optimal sigmas
optimal_sigma = result.x[0]

# Evaluate final objective
print("Evaluating final MSE...")
final_mse, _ = objective_and_constraint(optimal_sigma)
pop.set_sigmas(optimal_sigma)
final_rate = np.sum(np.mean(pop.compute_rates(itd_grid), axis=1))

print(f"Optimal sigma: {optimal_sigma}")
print(f"Final MSE: {final_mse:.2f} μs²")
print(f"Final RMSE: {np.sqrt(final_mse):.2f} μs")
print(f"Final total rate: {final_rate:.2f} Hz")
print(f"MSE improvement: {(1 - final_mse/initial_mse)*100:.2f}%")
print(f"RMSE improvement: {(1 - np.sqrt(final_mse)/np.sqrt(initial_mse))*100:.2f}%")
print()

# Display results
print("Neuron-by-neuron summary:")
print(f"{'Neuron':<8} {'Mean (μs)':<12} {'Sigma (μs)':<12} {'Avg Rate (Hz)':<15}")
print("-" * 50)
rates = pop.compute_rates(itd_grid)
for i in range(n_neurons):
    avg_rate = np.mean(rates[i, :])
    print(f"{i:<8} {pop.means[i]:<12.2f} {optimal_sigma:<12.2f} {avg_rate:<15.2f}")

print()
print(f"Total average rate: {final_rate:.2f} Hz (budget: {rate_budget} Hz)")
print(f"Decoding MSE: {final_mse:.2f} μs² (RMSE: {np.sqrt(final_mse):.2f} μs)")
