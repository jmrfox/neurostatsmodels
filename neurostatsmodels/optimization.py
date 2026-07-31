"""Resource-constrained optimization of population tuning parameters.

Efficient-coding toy problem: pick the shared tuning width of a
:class:`~neurostatsmodels.populations.GaussianTunedPopulation` that minimizes
decoding error, subject to a cap on the population's total average firing rate.
Narrow tuning raises peak rates and leaves gaps in stimulus coverage; broad
tuning wastes spikes on uninformative neurons, so an interior optimum exists.
"""

import numpy as np
from scipy.optimize import minimize


def total_average_rate(population, sigma, stimulus_grid):
    """Population-summed, stimulus-averaged firing rate at a tuning width.

    This is the metabolic cost term: ``sum_i mean_s r_i(s)``. It depends only on
    the tuning curves, so it is cheap compared to spike-based decoding.

    Parameters
    ----------
    population : GaussianTunedPopulation
        Population with means (and spontaneous rates) already set. Its sigmas
        are overwritten with ``sigma``.
    sigma : float
        Shared tuning width to evaluate.
    stimulus_grid : array-like
        Stimulus values to average the rates over.

    Returns
    -------
    float
        Total average firing rate across the population, in Hz.
    """
    population.set_sigmas(float(sigma))
    return float(np.sum(np.mean(population.compute_rates(stimulus_grid), axis=1)))


def evaluate_tuning_width(
    population,
    sigma,
    stimulus_grid,
    n_test_stimuli=8,
    n_trials_per_stim=20,
    trial_duration=1.0,
    random_seed=42,
):
    """Decoding RMSE and metabolic cost for one shared tuning width.

    Sets all neurons to ``sigma``, samples test stimuli uniformly from the grid,
    generates Poisson spikes, decodes each trial with the population MLE, and
    reports the root-mean-squared decoding error together with the total average
    firing rate. Deterministic for a fixed ``random_seed``, which lets callers
    cache results across repeated optimizer queries.

    Parameters
    ----------
    population : GaussianTunedPopulation
        Population with means, spontaneous rates, and refractory periods set.
    sigma : float or array-like
        Shared tuning width. Array-like input uses the first element, for
        compatibility with scipy optimizers.
    stimulus_grid : array-like
        Stimulus grid; its endpoints define the decoding/search range.
    n_test_stimuli : int, optional
        Number of random test stimuli. Default is 8.
    n_trials_per_stim : int, optional
        Trials generated per test stimulus. Default is 20.
    trial_duration : float, optional
        Trial length in seconds. Default is 1.0.
    random_seed : int, optional
        Seed for stimulus sampling and spike generation. Default is 42.

    Returns
    -------
    rmse : float
        Root-mean-squared decoding error, in stimulus units.
    total_rate : float
        Total average firing rate across the population, in Hz.
    """
    sigma = float(np.ravel(sigma)[0])
    population.set_sigmas(sigma)
    population.build_tuning_curves(stimulus_grid)

    stimulus_range = (float(np.min(stimulus_grid)), float(np.max(stimulus_grid)))
    rng = np.random.default_rng(random_seed)
    test_stimuli = rng.uniform(stimulus_range[0], stimulus_range[1], n_test_stimuli)

    squared_errors = []
    for true_stimulus in test_stimuli:
        spikes = population.generate_spikes(
            stimulus=true_stimulus,
            duration=trial_duration,
            n_trials=n_trials_per_stim,
            random_state=rng,
        )
        decoded = population.decode_mle(spikes, stimulus_range=stimulus_range)
        squared_errors.append((np.asarray(decoded) - true_stimulus) ** 2)

    rmse = float(np.sqrt(np.mean(np.concatenate(squared_errors))))
    total_rate = float(
        np.sum(np.mean(population.compute_rates(stimulus_grid), axis=1))
    )
    return rmse, total_rate


def sweep_tuning_widths(population, sigmas, stimulus_grid, rate_budget, **eval_kwargs):
    """Evaluate a list of tuning widths against the rate budget.

    Parameters
    ----------
    population : GaussianTunedPopulation
        Population to evaluate; its sigmas are overwritten.
    sigmas : array-like
        Tuning widths to evaluate.
    stimulus_grid : array-like
        Stimulus grid passed to :func:`evaluate_tuning_width`.
    rate_budget : float
        Cap on total average firing rate, in Hz.
    **eval_kwargs
        Forwarded to :func:`evaluate_tuning_width`.

    Yields
    ------
    dict
        Keys ``sigma``, ``rmse``, ``total_rate``, ``slack`` (budget minus rate;
        nonnegative when feasible), and ``feasible``.
    """
    for sigma in np.asarray(sigmas, dtype=float):
        rmse, total_rate = evaluate_tuning_width(
            population, sigma, stimulus_grid, **eval_kwargs
        )
        yield {
            "sigma": float(sigma),
            "rmse": rmse,
            "total_rate": total_rate,
            "slack": float(rate_budget - total_rate),
            "feasible": bool(total_rate <= rate_budget),
        }


def optimize_tuning_width(
    population,
    stimulus_grid,
    rate_budget,
    sigma_init=50.0,
    sigma_bounds=(5.0, 200.0),
    method="COBYLA",
    maxiter=30,
    eval_kwargs=None,
    on_evaluation=None,
):
    """Minimize decoding RMSE over tuning width subject to a rate budget.

    Wraps :func:`scipy.optimize.minimize` with the inequality constraint
    ``rate_budget - total_rate >= 0``. Objective and constraint share one cached
    evaluation per width, so each optimizer step costs a single spike-generation
    pass instead of two.

    Parameters
    ----------
    population : GaussianTunedPopulation
        Population with means, spontaneous rates, and refractory periods set.
    stimulus_grid : array-like
        Stimulus grid passed to :func:`evaluate_tuning_width`.
    rate_budget : float
        Cap on total average firing rate, in Hz.
    sigma_init : float, optional
        Initial tuning width. Default is 50.0.
    sigma_bounds : tuple, optional
        ``(low, high)`` bounds on the width. Default is ``(5.0, 200.0)``.
    method : str, optional
        A scipy method supporting inequality constraints, e.g. ``'COBYLA'`` or
        ``'SLSQP'``. Default is ``'COBYLA'``.
    maxiter : int, optional
        Iteration cap handed to the optimizer. Default is 30.
    eval_kwargs : dict, optional
        Forwarded to :func:`evaluate_tuning_width`.
    on_evaluation : callable, optional
        Called with each new evaluation record as it is computed. Useful for
        progress reporting in notebooks.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Raw optimizer result; ``result.x[0]`` is the optimized width.
    trace : list of dict
        One record per distinct width evaluated, in evaluation order, with keys
        ``sigma``, ``rmse``, ``total_rate``, ``slack``, and ``feasible``.
    """
    eval_kwargs = dict(eval_kwargs or {})
    low, high = float(sigma_bounds[0]), float(sigma_bounds[1])
    cache = {}
    trace = []

    def evaluate(sigma):
        width = float(np.clip(np.ravel(sigma)[0], low, high))
        key = round(width, 6)
        if key not in cache:
            rmse, total_rate = evaluate_tuning_width(
                population, width, stimulus_grid, **eval_kwargs
            )
            cache[key] = (rmse, total_rate)
            record = {
                "sigma": width,
                "rmse": rmse,
                "total_rate": total_rate,
                "slack": float(rate_budget - total_rate),
                "feasible": bool(total_rate <= rate_budget),
            }
            trace.append(record)
            if on_evaluation is not None:
                on_evaluation(record)
        return cache[key]

    result = minimize(
        lambda sigma: evaluate(sigma)[0],
        x0=[float(sigma_init)],
        method=method,
        bounds=[(low, high)],
        constraints=[{"type": "ineq", "fun": lambda s: rate_budget - evaluate(s)[1]}],
        options={"maxiter": int(maxiter)},
    )
    return result, trace
