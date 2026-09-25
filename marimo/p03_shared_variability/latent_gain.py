"""Project 3: shared variability via trial-wise latent gain modulation."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from neurostatsmodels.populations import GaussianTunedPopulation

    np.set_printoptions(precision=4, suppress=True, linewidth=1000)
    return GaussianTunedPopulation, go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Shared variability via latent gain

    Interactive first pass of Project 3. Each trial draws a shared multiplicative
    gain $g_t$ that scales every neuron's tuning curve:

    $$ r_i(s, t) = g_t \cdot f_i(s),\qquad n_i \mid g_t,s \sim \mathrm{Poisson}(r_i\,T) $$

    Questions we explore:

    - How does shared gain affect decoding precision?
    - Can $g_t$ be recovered from spike counts?
    - When does this variability hurt (vs help) coding?

    Base tuning $f_i(s)$ reuses `GaussianTunedPopulation` from Project 1. Trial
    gains and count simulation live as helpers in this notebook (not the library).
    Distinct from within-trial `alpha` suppression in `generate_spikes`.
    """)
    return


@app.cell
def _(np):
    def sample_trial_gains(n_trials, gain_cv, rng):
        """Draw positive trial gains with mean ≈ 1 and given coefficient of variation.

        Uses a lognormal with E[g]=1. When ``gain_cv`` is 0, returns ones (independent
        Poisson / no shared gain).
        """
        if gain_cv <= 0:
            return np.ones(n_trials, dtype=float)
        sigma = np.sqrt(np.log(1.0 + gain_cv**2))
        mu = -0.5 * sigma**2
        return rng.lognormal(mean=mu, sigma=sigma, size=n_trials)

    def simulate_gain_modulated_counts(base_rates, gains, duration, rng):
        """Poisson counts with trial-wise multiplicative gain.

        Parameters
        ----------
        base_rates : ndarray, shape (n_neurons,)
            Tuning rates f_i(s) in Hz at the true stimulus.
        gains : ndarray, shape (n_trials,)
            Trial gains g_t.
        duration : float
            Trial length T in seconds.
        rng : np.random.Generator

        Returns
        -------
        counts : ndarray, shape (n_trials, n_neurons)
        """
        lam = gains[:, None] * base_rates[None, :] * duration
        return rng.poisson(lam)

    def noise_correlation_matrix(counts):
        """Pearson correlation across trials; diagonal set to nan."""
        corr = np.corrcoef(counts, rowvar=False)
        np.fill_diagonal(corr, np.nan)
        return corr

    def mean_pairwise_rho(corr):
        """Mean of upper-triangle off-diagonal correlations."""
        iu = np.triu_indices_from(corr, k=1)
        return float(np.nanmean(corr[iu]))

    def fano_factors(counts):
        """Per-neuron Fano factor Var(n)/E[n] across trials."""
        means = np.mean(counts, axis=0)
        vars_ = np.var(counts, axis=0, ddof=1)
        return vars_ / np.maximum(means, 1e-12)

    def decode_mle_counts(counts, rate_grid, stim_grid, duration, gains=None):
        """Count-based Poisson MLE over a stimulus grid.

        ``rate_grid`` has shape (n_neurons, n_stim). If ``gains`` is given
        (length n_trials), each trial's likelihood uses g_t * f_i(s).
        """
        n_trials = counts.shape[0]
        s_hat = np.zeros(n_trials)
        log_rates = np.log(rate_grid + 1e-10)
        sum_rates = np.sum(rate_grid, axis=0)
        for t in range(n_trials):
            g = 1.0 if gains is None else float(gains[t])
            # log L(s) = sum_i [n_i log(g f_i) - g f_i T]
            ll = counts[t] @ log_rates - g * duration * sum_rates
            if gains is not None:
                ll = ll + np.sum(counts[t]) * np.log(g + 1e-10)
            s_hat[t] = stim_grid[np.argmax(ll)]
        return s_hat

    def decode_mle_profile_gain(counts, rate_grid, stim_grid, duration):
        """Joint MLE of stimulus with gain profiled out per candidate s.

        For each s, g_hat(s) = sum_i n_i / (T sum_i f_i(s)), then plug into
        the Poisson log-likelihood. Does not require knowing the true stimulus.
        """
        n_trials = counts.shape[0]
        s_hat = np.zeros(n_trials)
        log_rates = np.log(rate_grid + 1e-10)
        sum_rates = np.sum(rate_grid, axis=0)
        for t in range(n_trials):
            n_tot = float(np.sum(counts[t]))
            g_hat = n_tot / np.maximum(duration * sum_rates, 1e-12)
            ll = (
                counts[t] @ log_rates
                + n_tot * np.log(g_hat + 1e-10)
                - g_hat * duration * sum_rates
            )
            s_hat[t] = stim_grid[np.argmax(ll)]
        return s_hat

    def infer_shared_gain(counts, base_rates, duration):
        """Closed-form MLE of trial-wise shared gain given known f_i(s).

        For n_i ~ Poisson(g * f_i * T), the MLE is
        g_hat = sum_i n_i / (T * sum_i f_i).
        """
        denom = duration * np.sum(base_rates)
        return np.sum(counts, axis=1) / np.maximum(denom, 1e-12)

    return (
        decode_mle_counts,
        decode_mle_profile_gain,
        fano_factors,
        infer_shared_gain,
        mean_pairwise_rho,
        noise_correlation_matrix,
        sample_trial_gains,
        simulate_gain_modulated_counts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup and generate

    Choose population size, tuning width, true ITD, and the **gain CV**
    (0 = independent Poisson). Counts are drawn as
    $n_i \sim \mathrm{Poisson}(g_t\,f_i(s)\,T)$ — fast count simulation, not
    spike-time stepping.
    """)
    return


@app.cell
def _(mo):
    n_neurons = mo.ui.slider(4, 40, value=12, step=1, label="Neurons")
    sigma = mo.ui.slider(10.0, 100.0, value=30.0, step=5.0, label="σ (µs)")
    true_itd = mo.ui.slider(-200.0, 200.0, value=50.0, step=5.0, label="True ITD (µs)")
    n_trials = mo.ui.slider(50, 500, value=200, step=25, label="Trials")
    duration = mo.ui.slider(0.25, 3.0, value=1.0, step=0.25, label="Duration T (s)")
    gain_cv = mo.ui.slider(0.0, 1.5, value=1.0, step=0.05, label="Gain CV")
    sim_seed = mo.ui.number(0, 9999, value=42, step=1, label="Seed")
    run_sim = mo.ui.run_button(label="Generate trials")
    mo.vstack(
        [
            mo.hstack([n_neurons, sigma, true_itd], wrap=True),
            mo.hstack([n_trials, duration, gain_cv, sim_seed], wrap=True),
            run_sim,
        ]
    )
    return (
        duration,
        gain_cv,
        n_neurons,
        n_trials,
        run_sim,
        sigma,
        sim_seed,
        true_itd,
    )


@app.cell
def _(
    GaussianTunedPopulation,
    duration,
    gain_cv,
    mo,
    n_neurons,
    n_trials,
    np,
    run_sim,
    sample_trial_gains,
    sigma,
    sim_seed,
    simulate_gain_modulated_counts,
    true_itd,
):
    mo.stop(not run_sim.value, mo.md("_Click **Generate trials** to sample $g_t$ and counts._"))

    stim_grid = np.linspace(-200.0, 200.0, 401)
    pop = GaussianTunedPopulation(
        n_neurons=int(n_neurons.value),
        reference_rate=50.0,
        reference_sigma=30.0,
    )
    pop.set_means_uniform(stim_grid)
    pop.set_sigmas(float(sigma.value))
    pop.set_spontaneous_rates(2.0)

    true_s = float(true_itd.value)
    T = float(duration.value)
    base_rates = pop.compute_rates(true_s)  # (n_neurons,)

    rng = np.random.default_rng(int(sim_seed.value))
    gains = sample_trial_gains(int(n_trials.value), float(gain_cv.value), rng)
    counts = simulate_gain_modulated_counts(base_rates, gains, T, rng)
    rate_grid = pop.compute_rates(stim_grid)  # (n_neurons, n_stim)
    return T, base_rates, counts, gains, rate_grid, stim_grid, true_s


@app.cell
def _(base_rates, counts, gains, go, make_subplots, np, true_s):
    fig_gen = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Trial gains g_t (mean={np.mean(gains):.2f})", "Mean rates at true ITD"),
        horizontal_spacing=0.1,
    )
    fig_gen.add_trace(
        go.Histogram(
            x=gains,
            nbinsx=max(int(np.sqrt(len(gains))), 8),
            name="g_t",
            marker_color="#4C78A8",
            opacity=0.85,
        ),
        row=1,
        col=1,
    )
    fig_gen.add_trace(
        go.Bar(
            x=list(range(len(base_rates))),
            y=base_rates,
            name="f_i(s)",
            marker_color="#54A24B",
        ),
        row=1,
        col=2,
    )
    fig_gen.update_xaxes(title_text="g", row=1, col=1)
    fig_gen.update_xaxes(title_text="Neuron", row=1, col=2)
    fig_gen.update_yaxes(title_text="Count", row=1, col=1)
    fig_gen.update_yaxes(title_text="Rate (Hz)", row=1, col=2)
    fig_gen.update_layout(
        height=360,
        title_text=f"True ITD = {true_s:.0f} µs  |  mean count / neuron = {np.mean(counts):.1f}",
        showlegend=False,
        margin=dict(t=70, b=40),
    )
    fig_gen
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared variability signatures

    Shared $g_t$ induces **positive noise correlations** and **overdispersion**
    (Fano $> 1$) without changing mean tuning. Drag gain CV above and
    re-generate, or sweep CV below to see pairwise $\rho$ grow with shared gain.
    """)
    return


@app.cell
def _(
    counts,
    fano_factors,
    go,
    make_subplots,
    mean_pairwise_rho,
    mo,
    noise_correlation_matrix,
    np,
):
    corr = noise_correlation_matrix(counts)
    rho_mean = mean_pairwise_rho(corr)
    fanos = fano_factors(counts)

    fig_var = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Noise correlation (mean ρ = {rho_mean:.3f})",
            "Fano factors (Poisson → 1)",
        ),
        horizontal_spacing=0.12,
        column_widths=[0.55, 0.45],
    )
    fig_var.add_trace(
        go.Heatmap(
            z=corr,
            colorscale="RdBu_r",
            zmid=0.0,
            colorbar=dict(title="ρ", x=0.46, thickness=12),
            hovertemplate="i=%{x}, j=%{y}<br>ρ=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_var.add_trace(
        go.Bar(
            x=list(range(len(fanos))),
            y=fanos,
            marker_color="#F58518",
            name="Fano",
            hovertemplate="neuron %{x}<br>F=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_var.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
    fig_var.update_xaxes(title_text="Neuron", row=1, col=1)
    fig_var.update_yaxes(title_text="Neuron", row=1, col=1, autorange="reversed")
    fig_var.update_xaxes(title_text="Neuron", row=1, col=2)
    fig_var.update_yaxes(title_text="Fano", rangemode="tozero", row=1, col=2)
    fig_var.update_layout(height=400, showlegend=False, margin=dict(t=60, b=40))

    var_note = mo.md(
        f"Mean pairwise ρ = **{rho_mean:.3f}**; mean Fano = **{np.mean(fanos):.2f}** "
        f"(independent Poisson would give ρ ≈ 0 and Fano ≈ 1)."
    )
    mo.vstack([fig_var, var_note])
    return


@app.cell
def _(mo):
    sweep_cv_max = mo.ui.slider(0.2, 1.5, value=1.0, step=0.1, label="Max CV in sweep")
    sweep_cv_points = mo.ui.slider(5, 16, value=8, step=1, label="Sweep points")
    run_cv_sweep = mo.ui.run_button(label="Sweep gain CV → ρ, Fano")
    mo.hstack([sweep_cv_max, sweep_cv_points, run_cv_sweep], wrap=True)
    return run_cv_sweep, sweep_cv_max, sweep_cv_points


@app.cell
def _(
    T,
    base_rates,
    fano_factors,
    mean_pairwise_rho,
    mo,
    noise_correlation_matrix,
    np,
    run_cv_sweep,
    sample_trial_gains,
    sim_seed,
    simulate_gain_modulated_counts,
    sweep_cv_max,
    sweep_cv_points,
):
    mo.stop(
        not run_cv_sweep.value,
        mo.md("_Click **Sweep gain CV → ρ, Fano** to map shared gain onto correlations._"),
    )

    cv_grid = np.linspace(0.0, float(sweep_cv_max.value), int(sweep_cv_points.value))
    rho_vs_cv = []
    fano_vs_cv = []
    with mo.status.progress_bar(
        total=len(cv_grid), title="Sweeping gain CV", subtitle="resampling counts"
    ) as bar:
        for cv in cv_grid:
            rng_s = np.random.default_rng(int(sim_seed.value) + int(1000 * cv))
            g_s = sample_trial_gains(200, float(cv), rng_s)
            c_s = simulate_gain_modulated_counts(base_rates, g_s, T, rng_s)
            rho_vs_cv.append(mean_pairwise_rho(noise_correlation_matrix(c_s)))
            fano_vs_cv.append(float(np.mean(fano_factors(c_s))))
            bar.update()
    return cv_grid, fano_vs_cv, rho_vs_cv


@app.cell
def _(cv_grid, fano_vs_cv, go, make_subplots, rho_vs_cv):
    fig_sweep = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sweep.add_trace(
        go.Scatter(
            x=cv_grid,
            y=rho_vs_cv,
            mode="lines+markers",
            name="Mean pairwise ρ",
            line=dict(color="#4C78A8", width=2.5),
        ),
        secondary_y=False,
    )
    fig_sweep.add_trace(
        go.Scatter(
            x=cv_grid,
            y=fano_vs_cv,
            mode="lines+markers",
            name="Mean Fano",
            line=dict(color="#F58518", width=2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig_sweep.add_hline(y=1.0, line_dash="dash", line_color="gray", secondary_y=True)
    fig_sweep.update_xaxes(title_text="Gain CV")
    fig_sweep.update_yaxes(title_text="Mean pairwise ρ", rangemode="tozero", secondary_y=False)
    fig_sweep.update_yaxes(title_text="Mean Fano", rangemode="tozero", secondary_y=True)
    fig_sweep.update_layout(
        height=380,
        title_text="Shared gain → correlations and overdispersion",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60, b=40),
        hovermode="x unified",
    )
    fig_sweep
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decoding with and without $g_t$

    Independent-Poisson MLE **ignores** shared gain. Conditioning on oracle
    $g_t$ restores the correct likelihood. **Profile** $\hat g(s)$ maximizes
    jointly over stimulus and gain without knowing the true ITD. Unmodeled
    shared gain typically **hurts** precision — correlations break naive
    Fisher-information additivity.
    """)
    return


@app.cell
def _(
    T,
    base_rates,
    counts,
    decode_mle_counts,
    decode_mle_profile_gain,
    gains,
    go,
    infer_shared_gain,
    mo,
    np,
    rate_grid,
    stim_grid,
    true_s,
):
    gains_hat = infer_shared_gain(counts, base_rates, T)

    s_ignore = decode_mle_counts(counts, rate_grid, stim_grid, T, gains=None)
    s_oracle = decode_mle_counts(counts, rate_grid, stim_grid, T, gains=gains)
    s_profile = decode_mle_profile_gain(counts, rate_grid, stim_grid, T)

    def _rmse(s_hat):
        return float(np.sqrt(np.mean((s_hat - true_s) ** 2)))

    rmse_ignore = _rmse(s_ignore)
    rmse_oracle = _rmse(s_oracle)
    rmse_profile = _rmse(s_profile)

    n_bins = max(int(np.sqrt(len(s_ignore))), 8)
    fig_dec = go.Figure()
    for s_hat, name, color in (
        (s_ignore, f"Ignore g  (RMSE {rmse_ignore:.1f})", "#E45756"),
        (s_oracle, f"Oracle g  (RMSE {rmse_oracle:.1f})", "#54A24B"),
        (s_profile, f"Profile ĝ(s)  (RMSE {rmse_profile:.1f})", "#4C78A8"),
    ):
        fig_dec.add_trace(
            go.Histogram(
                x=s_hat,
                nbinsx=n_bins,
                name=name,
                opacity=0.55,
                marker_color=color,
            )
        )
    fig_dec.add_vline(x=true_s, line_dash="dash", line_color="black")
    fig_dec.update_layout(
        height=400,
        barmode="overlay",
        title_text=f"MLE estimates (true ITD = {true_s:.0f} µs)",
        xaxis_title="Decoded ITD (µs)",
        yaxis_title="Count",
        xaxis=dict(range=[-200.0, 200.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=70, b=40),
    )

    dec_stats = mo.hstack(
        [
            mo.stat(f"{rmse_ignore:.1f} µs", label="RMSE ignore g", bordered=True),
            mo.stat(f"{rmse_oracle:.1f} µs", label="RMSE oracle g", bordered=True),
            mo.stat(f"{rmse_profile:.1f} µs", label="RMSE profile ĝ(s)", bordered=True),
        ],
        wrap=True,
    )
    mo.vstack([fig_dec, dec_stats])
    return (gains_hat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inferring the latent

    Given known tuning $f_i(s)$ at the true stimulus, the shared-gain MLE is
    closed form:

    $$ \hat g_t = \frac{\sum_i n_{i,t}}{T \sum_i f_i(s)} $$

    This is a **shared-gain** estimate, not full Poisson factor analysis (one
    loading per neuron, EM, etc.).
    """)
    return


@app.cell
def _(gains, gains_hat, go, mo, np):
    gain_corr = float(np.corrcoef(gains, gains_hat)[0, 1])
    gain_rmse = float(np.sqrt(np.mean((gains_hat - gains) ** 2)))

    fig_gain = go.Figure()
    fig_gain.add_trace(
        go.Scatter(
            x=gains,
            y=gains_hat,
            mode="markers",
            name="trials",
            marker=dict(size=7, color="#4C78A8", opacity=0.65),
            hovertemplate="true=%{x:.3f}<br>ĝ=%{y:.3f}<extra></extra>",
        )
    )
    lims = [
        float(min(gains.min(), gains_hat.min())),
        float(max(gains.max(), gains_hat.max())),
    ]
    fig_gain.add_trace(
        go.Scatter(
            x=lims,
            y=lims,
            mode="lines",
            name="identity",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig_gain.update_layout(
        height=400,
        title_text=f"Recovered vs true g_t  |  corr={gain_corr:.3f}  |  RMSE={gain_rmse:.3f}",
        xaxis_title="True g_t",
        yaxis_title="Inferred ĝ_t",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60, b=40),
    )

    gain_stats = mo.hstack(
        [
            mo.stat(f"{gain_corr:.3f}", label="corr(g, ĝ)", bordered=True),
            mo.stat(f"{gain_rmse:.3f}", label="RMSE(ĝ)", bordered=True),
        ],
        wrap=True,
    )
    mo.vstack([fig_gain, gain_stats])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Takeaways

    - A single trial-wise latent $g_t$ creates **structured** noise: positive
      correlations and Fano $> 1$, even though neurons are conditionally
      independent given $g_t$.
    - Ignoring shared gain in the decoder **hurts** ITD estimates; conditioning
      on oracle $g_t$ or profiling $\hat g(s)$ recovers performance.
    - With known tuning at the true stimulus, $\hat g_t$ is a one-line MLE —
      enough to see that the latent is recoverable. Full Poisson FA / demixed
      PCA are natural next steps (and Project 3.5 adds heterogeneous noise).
    """)
    return


if __name__ == "__main__":
    app.run()
