"""Interactive first pass of Project 1: Gaussian ITD tuning, FI, and MLE decoding."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from wigglystuff import CopyToClipboard, Slider2D

    from neurostatsmodels.populations import GaussianTunedPopulation
    from neurostatsmodels.optimization import (
        optimize_tuning_width,
        sweep_tuning_widths,
        total_average_rate,
    )

    np.set_printoptions(precision=4, suppress=True, linewidth=1000)
    return (
        CopyToClipboard,
        GaussianTunedPopulation,
        Slider2D,
        go,
        make_subplots,
        mo,
        np,
        optimize_tuning_width,
        sweep_tuning_widths,
        total_average_rate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic model of ITD-tuned populations

    Interactive first pass of Project 1: Gaussian ITD tuning, population Fisher
    information, spike generation, and MLE decoding.

    - Neurons have Gaussian ITD tuning with shared width $\sigma$ and evenly spaced
      preferred ITDs on $[-200, 200]$ µs
    - Responses are independent Poisson (with refractory period for spikes)
    - Decoder recovers ITD via maximum likelihood
    - Finally, $\sigma$ is optimized for decoding accuracy under a population
      firing-rate budget

    Plots use **Plotly** (hover, zoom, toggle traces).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tuning curves and Fisher information

    Rate of neuron $i$:

    $$ r_i(s) = r_0 + r_{\text{max}} \exp\left(-\frac{(s - \mu_i)^2}{2\sigma^2}\right) $$

    with peak-rate normalization $r_{\text{max}} = r_{\text{ref}}\,\sigma_{\text{ref}} / \sigma$.

    Single-neuron FI (per unit time) for a Poisson neuron:

    $$ \frac{J_i(s)}{\Delta t} = \frac{\left( \frac{dr_i}{ds} \right)^2}{r_i(s)} $$

    Population FI is the sum across neurons. Units: ITD in µs, rate in Hz,
    so $J/\Delta t$ has units µs$^{-2}$.
    """)
    return


@app.cell
def _(Slider2D, mo):
    n_neurons = mo.ui.slider(2, 40, value=10, step=1, label="Neurons")
    spontaneous = mo.ui.slider(0.0, 10.0, value=2.0, step=0.5, label="Spontaneous rate (Hz)")
    # x = query ITD (µs), y = shared tuning width σ (µs)
    itd_sigma = mo.ui.anywidget(
        Slider2D(
            x=50.0,
            y=30.0,
            width=280,
            height=280,
            x_bounds=(-200.0, 200.0),
            y_bounds=(5.0, 120.0),
        )
    )
    mo.vstack(
        [
            mo.hstack([n_neurons, spontaneous], wrap=True),
            mo.md("Drag the pad: **x** = query ITD (µs), **y** = σ (µs)"),
            itd_sigma,
        ]
    )
    return itd_sigma, n_neurons, spontaneous


@app.cell
def _(GaussianTunedPopulation, itd_sigma, n_neurons, np, spontaneous):
    query_itd = float(itd_sigma.x)
    sigma = float(itd_sigma.y)
    stimulus_grid = np.linspace(-200.0, 200.0, 401)
    pop = GaussianTunedPopulation(
        n_neurons=int(n_neurons.value),
        reference_rate=50.0,
        reference_sigma=30.0,
    )
    pop.set_means_uniform(stimulus_grid)
    pop.set_sigmas(sigma)
    pop.set_spontaneous_rates(float(spontaneous.value))
    pop.build_tuning_curves(stimulus_grid)
    pop.build_fisher_info_curves(epsilon=1e-10)

    query_rates = pop.get_rates_at(query_itd)
    query_fi = pop.get_population_fisher_info_at(query_itd)
    return pop, query_fi, query_itd, query_rates, sigma, stimulus_grid


@app.cell
def _(
    CopyToClipboard,
    go,
    make_subplots,
    mo,
    n_neurons,
    np,
    pop,
    query_fi,
    query_itd,
    query_rates,
    sigma,
    stimulus_grid,
):
    n = int(n_neurons.value)
    rates = pop.tuning_curves.values
    fi_curves = pop.fisher_info_curves.values
    pop_fi_curve = np.sum(fi_curves, axis=1)

    fig_fi = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Tuning curves",
            "Individual Fisher information",
            "Population Fisher information",
        ),
    )

    for i in range(n):
        name = f"n{i}"
        hover = "ITD=%{x:.1f} µs<br>rate=%{y:.2f} Hz<extra>" + name + "</extra>"
        fig_fi.add_trace(
            go.Scatter(
                x=stimulus_grid,
                y=rates[:, i],
                mode="lines",
                name=name,
                legendgroup=name,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )
        fig_fi.add_trace(
            go.Scatter(
                x=stimulus_grid,
                y=fi_curves[:, i],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                hovertemplate="ITD=%{x:.1f} µs<br>FI=%{y:.4f}<extra>" + name + "</extra>",
            ),
            row=2,
            col=1,
        )

    fig_fi.add_trace(
        go.Scatter(
            x=stimulus_grid,
            y=pop_fi_curve,
            mode="lines",
            name="population",
            line=dict(color="black", width=2.5),
            hovertemplate="ITD=%{x:.1f} µs<br>pop FI=%{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    for row in (1, 2, 3):
        fig_fi.add_vline(
            x=query_itd,
            line_dash="dash",
            line_color="gray",
            opacity=0.7,
            row=row,
            col=1,
        )

    fig_fi.update_yaxes(title_text="Rate (Hz)", rangemode="tozero", row=1, col=1)
    fig_fi.update_yaxes(title_text="FI / time", rangemode="tozero", row=2, col=1)
    fig_fi.update_yaxes(title_text="Pop. FI / time", rangemode="tozero", row=3, col=1)
    fig_fi.update_xaxes(title_text="ITD (µs)", row=3, col=1)
    fig_fi.update_layout(
        height=780,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )

    summary = (
        f"At ITD = {query_itd:.1f} µs, σ = {sigma:.1f} µs: "
        f"population FI = {query_fi:.4f}; "
        f"rates = {np.array2string(query_rates, precision=2)}"
    )
    copy_summary = mo.ui.anywidget(CopyToClipboard(summary))
    mo.vstack([fig_fi, mo.md(summary), copy_summary])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spike generation

    Encode a fixed ITD as spike times (absolute refractory period included).
    Hover a tick for neuron index and time; rates should match the tuning curves
    above at the chosen ITD.
    """)
    return


@app.cell
def _(mo):
    spike_itd = mo.ui.slider(-200.0, 200.0, value=50.0, step=5.0, label="Encoded ITD (µs)")
    spike_duration = mo.ui.slider(0.5, 5.0, value=2.0, step=0.5, label="Duration (s)")
    refractory_ms = mo.ui.slider(0.0, 20.0, value=5.0, step=1.0, label="Refractory (ms)")
    mo.hstack([spike_itd, spike_duration, refractory_ms], wrap=True)
    return refractory_ms, spike_duration, spike_itd


@app.cell
def _(
    GaussianTunedPopulation,
    go,
    n_neurons,
    np,
    refractory_ms,
    sigma,
    spike_duration,
    spike_itd,
    spontaneous,
):
    pop_spikes = GaussianTunedPopulation(
        n_neurons=int(n_neurons.value),
        reference_rate=50.0,
        reference_sigma=30.0,
    )
    pop_spikes.set_means_uniform(np.linspace(-200.0, 200.0, 401))
    pop_spikes.set_sigmas(sigma)
    pop_spikes.set_spontaneous_rates(float(spontaneous.value))
    pop_spikes.set_refractory_periods(float(refractory_ms.value) / 1000.0)

    spikes = pop_spikes.generate_spikes(
        float(spike_itd.value),
        duration=float(spike_duration.value),
        n_trials=1,
        dt=0.001,
        alpha=0.0,
        random_state=0,
    )

    trial_interval = spikes.time_support[0]
    trial_start = float(trial_interval.start[0])
    trial_end = float(trial_interval.end[0])
    neuron_indices = sorted(spikes.keys())

    raster_x = []
    raster_y = []
    raster_text = []
    for plot_idx, neuron_idx in enumerate(neuron_indices):
        trial_spikes = spikes[neuron_idx].restrict(trial_interval)
        if len(trial_spikes) == 0:
            continue
        times = trial_spikes.t - trial_start
        raster_x.extend(times.tolist())
        raster_y.extend([plot_idx] * len(times))
        raster_text.extend(
            [f"neuron {neuron_idx}<br>t = {t:.3f} s" for t in times]
        )

    fig_raster = go.Figure(
        data=[
            go.Scatter(
                x=raster_x,
                y=raster_y,
                mode="markers",
                marker=dict(
                    symbol="line-ns",
                    size=12,
                    line=dict(width=1.5, color="black"),
                    color="black",
                ),
                text=raster_text,
                hovertemplate="%{text}<extra></extra>",
                name="spikes",
            )
        ]
    )
    fig_raster.update_layout(
        title=f"Spikes at ITD = {float(spike_itd.value):.0f} µs",
        xaxis_title="Time (s)",
        yaxis_title="Neuron",
        height=max(280, 40 + 28 * len(neuron_indices)),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(neuron_indices))),
            ticktext=[str(i) for i in neuron_indices],
            autorange="reversed",
            range=[-0.5, len(neuron_indices) - 0.5],
        ),
        xaxis=dict(range=[0, trial_end - trial_start]),
        margin=dict(t=50, b=40),
        showlegend=False,
    )
    fig_raster
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MLE decoding

    Generate many trials at a true ITD, then decode each trial with the
    population MLE. Brush-select histogram bins to inspect a subset of estimates
    (Plotly selection is reactive via `mo.ui.plotly`).
    """)
    return


@app.cell
def _(mo):
    true_itd = mo.ui.slider(-200.0, 200.0, value=50.0, step=5.0, label="True ITD (µs)")
    n_trials = mo.ui.slider(20, 200, value=100, step=10, label="Trials")
    trial_duration = mo.ui.slider(0.25, 2.0, value=1.0, step=0.25, label="Trial duration (s)")
    run_decode = mo.ui.run_button(label="Run MLE decode")
    mo.hstack([true_itd, n_trials, trial_duration, run_decode], wrap=True)
    return n_trials, run_decode, trial_duration, true_itd


@app.cell
def _(
    GaussianTunedPopulation,
    go,
    mo,
    n_neurons,
    n_trials,
    np,
    run_decode,
    sigma,
    spontaneous,
    trial_duration,
    true_itd,
):
    mo.stop(not run_decode.value, mo.md("_Click **Run MLE decode** to generate trials._"))

    pop_decode = GaussianTunedPopulation(
        n_neurons=int(n_neurons.value),
        reference_rate=50.0,
        reference_sigma=30.0,
    )
    pop_decode.set_means_uniform(np.linspace(-200.0, 200.0, 401))
    pop_decode.set_sigmas(sigma)
    pop_decode.set_spontaneous_rates(float(spontaneous.value))
    pop_decode.set_refractory_periods(0.01)

    true_s = float(true_itd.value)
    decode_spikes = pop_decode.generate_spikes(
        true_s,
        duration=float(trial_duration.value),
        n_trials=int(n_trials.value),
        random_state=42,
    )
    s_hat = pop_decode.decode_mle(decode_spikes, stimulus_range=(-200.0, 200.0))

    bias = float(np.mean(s_hat) - true_s)
    std = float(np.std(s_hat))
    mae = float(np.mean(np.abs(s_hat - true_s)))
    mean_hat = float(np.mean(s_hat))
    n_bins = max(int(np.sqrt(len(s_hat))), 5)

    fig_hist = go.Figure(
        data=[
            go.Histogram(
                x=s_hat,
                nbinsx=n_bins,
                name="MLE estimates",
                marker_color="#4C78A8",
                opacity=0.85,
            )
        ]
    )
    for x_val, color, label, dash in (
        (true_s, "#E45756", "True", "dash"),
        (mean_hat, "#333333", "Mean MLE", "solid"),
    ):
        fig_hist.add_shape(
            type="line",
            x0=x_val,
            x1=x_val,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=color, dash=dash, width=2),
        )
        # Legend entry only (vline shapes are not listed in the legend)
        fig_hist.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                line=dict(color=color, dash=dash, width=2),
            )
        )
    fig_hist.update_layout(
        height=420,
        title_text=(
            f"Mean {mean_hat:.2f} µs  |  bias {bias:.2f}  |  "
            f"std {std:.2f}  |  MAE {mae:.2f}"
        ),
        xaxis_title="Decoded ITD (µs)",
        yaxis_title="Count",
        xaxis=dict(range=[-200.0, 200.0]),
        bargap=0.05,
        margin=dict(t=70, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    hist_chart = mo.ui.plotly(fig_hist)
    hist_chart
    return (hist_chart,)


@app.cell
def _(hist_chart, mo):
    selected = hist_chart.value
    selection_note = (
        mo.md(f"Selected points from brush: `{len(selected)}`")
        if selected
        else mo.md("_Brush histogram bars to inspect a subset of estimates._")
    )
    selection_note
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimal tuning width under a firing-rate budget

    Narrow tuning sharpens each neuron's slope but leaves gaps in stimulus
    coverage; broad tuning covers everything but spends spikes on uninformative
    neurons. With a cap on the population's metabolic cost there is an interior
    optimum.

    **Objective** (minimize): RMSE of the MLE decoder, estimated by sampling test
    ITDs, generating trials, and decoding each one.

    **Constraint** (feasible when $\ge 0$):

    $$ \text{slack}(\sigma) = R_{\text{budget}} - \sum_i \overline{r_i(s)} $$

    where $\overline{r_i(s)}$ averages neuron $i$'s rate over the ITD domain.

    The evaluation and optimizer live in
    [`neurostatsmodels/optimization.py`](../../neurostatsmodels/optimization.py);
    the cells below just drive them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Where is the budget feasible?

    The cost term needs no spikes, so this panel updates instantly. Note that
    the peak-rate normalization $r_{\max} = r_{\text{ref}}\sigma_{\text{ref}}/\sigma$
    roughly conserves area under each tuning curve, so cost is only weakly
    dependent on $\sigma$: the budget mostly limits **neuron count** and
    **spontaneous rate**, while $\sigma$ is set by the decoding term.
    """)
    return


@app.cell
def _(mo):
    opt_n_neurons = mo.ui.slider(2, 30, value=10, step=1, label="Neurons")
    opt_reference_rate = mo.ui.slider(
        10.0, 150.0, value=50.0, step=5.0, label="Reference peak rate (Hz)"
    )
    opt_spontaneous = mo.ui.slider(
        0.0, 10.0, value=3.0, step=0.5, label="Spontaneous rate (Hz)"
    )
    opt_refractory_ms = mo.ui.slider(
        0.0, 20.0, value=5.0, step=1.0, label="Refractory (ms)"
    )
    opt_budget = mo.ui.slider(
        50.0, 600.0, value=200.0, step=10.0, label="Rate budget (Hz)"
    )
    mo.hstack(
        [
            opt_n_neurons,
            opt_reference_rate,
            opt_spontaneous,
            opt_refractory_ms,
            opt_budget,
        ],
        wrap=True,
    )
    return (
        opt_budget,
        opt_n_neurons,
        opt_reference_rate,
        opt_refractory_ms,
        opt_spontaneous,
    )


@app.cell
def _(
    GaussianTunedPopulation,
    np,
    opt_n_neurons,
    opt_reference_rate,
    opt_refractory_ms,
    opt_spontaneous,
):
    opt_grid = np.linspace(-200.0, 200.0, 401)
    opt_sigma_limits = (5.0, 200.0)

    def make_opt_population():
        """Fresh population matching the current controls (sigmas set later)."""
        population = GaussianTunedPopulation(
            n_neurons=int(opt_n_neurons.value),
            reference_rate=float(opt_reference_rate.value),
            reference_sigma=30.0,
        )
        population.set_means_uniform(opt_grid)
        population.set_spontaneous_rates(float(opt_spontaneous.value))
        population.set_refractory_periods(float(opt_refractory_ms.value) / 1000.0)
        return population

    return make_opt_population, opt_grid, opt_sigma_limits


@app.cell
def _(
    go,
    make_opt_population,
    mo,
    np,
    opt_budget,
    opt_grid,
    opt_sigma_limits,
    total_average_rate,
):
    cost_sigmas = np.linspace(opt_sigma_limits[0], opt_sigma_limits[1], 100)
    cost_pop = make_opt_population()
    cost_rates = np.array(
        [total_average_rate(cost_pop, width, opt_grid) for width in cost_sigmas]
    )
    cost_budget = float(opt_budget.value)
    cost_feasible = cost_sigmas[cost_rates <= cost_budget]

    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Scatter(
            x=cost_sigmas,
            y=cost_rates,
            mode="lines",
            name="Total average rate",
            line=dict(color="#4C78A8", width=2.5),
            hovertemplate="σ=%{x:.1f} µs<br>cost=%{y:.1f} Hz<extra></extra>",
        )
    )
    fig_cost.add_trace(
        go.Scatter(
            x=cost_sigmas,
            y=np.full_like(cost_sigmas, cost_budget),
            mode="lines",
            name="Budget",
            line=dict(color="#E45756", width=2, dash="dash"),
            hovertemplate="budget=%{y:.1f} Hz<extra></extra>",
        )
    )
    fig_cost.add_hrect(
        y0=cost_budget,
        y1=max(cost_budget, float(cost_rates.max())) * 1.05 + 1.0,
        fillcolor="#E45756",
        opacity=0.08,
        line_width=0,
        annotation_text="infeasible",
        annotation_position="top left",
    )
    fig_cost.update_layout(
        height=340,
        title_text="Metabolic cost vs tuning width",
        xaxis_title="σ (µs)",
        yaxis_title="Total average rate (Hz)",
        yaxis=dict(rangemode="tozero"),
        margin=dict(t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    if cost_feasible.size == 0:
        cost_note = mo.md(
            "**No feasible σ** in this range — raise the budget, or lower the "
            "neuron count / reference rate / spontaneous rate."
        ).callout(kind="warn")
    elif cost_feasible.size == cost_sigmas.size:
        cost_note = mo.md(
            f"Every σ in [{opt_sigma_limits[0]:.0f}, {opt_sigma_limits[1]:.0f}] µs "
            f"is feasible (cost peaks at {cost_rates.max():.1f} Hz). The budget is "
            "not binding, so the optimum below is set purely by decoding accuracy."
        ).callout(kind="success")
    else:
        cost_note = mo.md(
            f"Feasible σ range: **{cost_feasible.min():.0f} – "
            f"{cost_feasible.max():.0f} µs** "
            f"(cost spans {cost_rates.min():.1f} – {cost_rates.max():.1f} Hz)."
        ).callout(kind="info")

    mo.vstack([fig_cost, cost_note])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Sweep σ

    Each σ costs one full encode/decode pass, so the fidelity sliders trade
    runtime against noise in the RMSE estimate. Start coarse, then refine around
    the minimum.
    """)
    return


@app.cell
def _(mo):
    sweep_range = mo.ui.range_slider(
        5.0,
        200.0,
        value=[10.0, 160.0],
        step=5.0,
        label="σ range (µs)",
        show_value=True,
    )
    sweep_points = mo.ui.slider(4, 24, value=8, step=1, label="Sweep points")
    sweep_stimuli = mo.ui.slider(2, 20, value=4, step=1, label="Test ITDs")
    sweep_trials = mo.ui.slider(5, 60, value=10, step=5, label="Trials / ITD")
    sweep_duration = mo.ui.slider(
        0.25, 3.0, value=0.5, step=0.25, label="Trial duration (s)"
    )
    sweep_seed = mo.ui.number(0, 9999, value=42, step=1, label="Seed")
    run_sweep = mo.ui.run_button(label="Run σ sweep")
    mo.vstack(
        [
            mo.hstack([sweep_range, sweep_points], wrap=True),
            mo.hstack(
                [sweep_stimuli, sweep_trials, sweep_duration, sweep_seed], wrap=True
            ),
            run_sweep,
        ]
    )
    return (
        run_sweep,
        sweep_duration,
        sweep_points,
        sweep_range,
        sweep_seed,
        sweep_stimuli,
        sweep_trials,
    )


@app.cell
def _(mo, sweep_duration, sweep_points, sweep_stimuli, sweep_trials):
    sweep_passes = int(sweep_points.value) * int(sweep_stimuli.value)
    sweep_sim_seconds = (
        sweep_passes * int(sweep_trials.value) * float(sweep_duration.value)
    )
    mo.md(
        f"Workload: **{sweep_passes} encode/decode passes** "
        f"({sweep_sim_seconds:.0f} s of simulated spiking)."
    )
    return


@app.cell
def _(
    make_opt_population,
    mo,
    np,
    opt_budget,
    opt_grid,
    run_sweep,
    sweep_duration,
    sweep_points,
    sweep_range,
    sweep_seed,
    sweep_stimuli,
    sweep_trials,
    sweep_tuning_widths,
):
    mo.stop(
        not run_sweep.value,
        mo.md("_Click **Run σ sweep** to evaluate decoding across tuning widths._"),
    )

    sweep_sigmas = np.linspace(
        float(sweep_range.value[0]),
        float(sweep_range.value[1]),
        int(sweep_points.value),
    )
    sweep_pop = make_opt_population()
    sweep_records = []
    with mo.status.progress_bar(
        total=len(sweep_sigmas),
        title="Sweeping σ",
        subtitle="encoding and MLE-decoding trials",
    ) as sweep_bar:
        for sweep_record in sweep_tuning_widths(
            sweep_pop,
            sweep_sigmas,
            opt_grid,
            float(opt_budget.value),
            n_test_stimuli=int(sweep_stimuli.value),
            n_trials_per_stim=int(sweep_trials.value),
            trial_duration=float(sweep_duration.value),
            random_seed=int(sweep_seed.value),
        ):
            sweep_records.append(sweep_record)
            sweep_bar.update()
    return (sweep_records,)


@app.cell
def _(go, make_subplots, mo, np, opt_budget, sweep_records):
    sweep_sigma_values = np.array([r["sigma"] for r in sweep_records])
    sweep_rmse = np.array([r["rmse"] for r in sweep_records])
    sweep_rates = np.array([r["total_rate"] for r in sweep_records])
    sweep_feasible = np.array([r["feasible"] for r in sweep_records])

    fig_sweep = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sweep.add_trace(
        go.Scatter(
            x=sweep_sigma_values,
            y=sweep_rmse,
            mode="lines+markers",
            name="Decoding RMSE",
            line=dict(color="#4C78A8", width=2.5),
            marker=dict(size=8),
            hovertemplate="σ=%{x:.1f} µs<br>RMSE=%{y:.2f} µs<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_sweep.add_trace(
        go.Scatter(
            x=sweep_sigma_values,
            y=sweep_rates,
            mode="lines",
            name="Total rate",
            line=dict(color="#54A24B", width=2, dash="dot"),
            hovertemplate="σ=%{x:.1f} µs<br>cost=%{y:.1f} Hz<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_sweep.add_trace(
        go.Scatter(
            x=sweep_sigma_values,
            y=np.full_like(sweep_sigma_values, float(opt_budget.value)),
            mode="lines",
            name="Budget",
            line=dict(color="#E45756", width=2, dash="dash"),
            hovertemplate="budget=%{y:.1f} Hz<extra></extra>",
        ),
        secondary_y=True,
    )

    if sweep_feasible.any():
        sweep_best_idx = int(np.argmin(np.where(sweep_feasible, sweep_rmse, np.inf)))
        fig_sweep.add_trace(
            go.Scatter(
                x=[sweep_sigma_values[sweep_best_idx]],
                y=[sweep_rmse[sweep_best_idx]],
                mode="markers",
                name="Best feasible σ",
                marker=dict(color="#B279A2", size=16, symbol="star"),
                hovertemplate="best σ=%{x:.1f} µs<br>RMSE=%{y:.2f} µs<extra></extra>",
            ),
            secondary_y=False,
        )
        sweep_note = mo.md(
            f"Best feasible σ on this grid: **{sweep_sigma_values[sweep_best_idx]:.1f} µs** "
            f"→ RMSE **{sweep_rmse[sweep_best_idx]:.2f} µs**, cost "
            f"{sweep_rates[sweep_best_idx]:.1f} Hz. Use it as the optimizer's "
            "starting point below."
        ).callout(kind="info")
    else:
        sweep_note = mo.md(
            "No swept σ satisfies the budget — raise the budget or shrink the population."
        ).callout(kind="warn")

    fig_sweep.update_yaxes(
        title_text="Decoding RMSE (µs)", rangemode="tozero", secondary_y=False
    )
    fig_sweep.update_yaxes(
        title_text="Total average rate (Hz)", rangemode="tozero", secondary_y=True
    )
    fig_sweep.update_layout(
        height=430,
        title_text="Objective and constraint vs tuning width",
        xaxis_title="σ (µs)",
        margin=dict(t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    mo.vstack([fig_sweep, sweep_note])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Constrained optimization

    COBYLA and SLSQP both handle the inequality constraint directly. Objective
    and constraint share a cached evaluation per σ, so each visited width costs
    one encode/decode pass rather than two. Evaluation fidelity is inherited
    from the sweep controls above; COBYLA spends the budget below on function
    evaluations, SLSQP on iterations.
    """)
    return


@app.cell
def _(mo):
    fit_method = mo.ui.dropdown(
        options=["COBYLA", "SLSQP"], value="COBYLA", label="Method"
    )
    fit_sigma_init = mo.ui.slider(5.0, 200.0, value=50.0, step=5.0, label="Initial σ (µs)")
    fit_maxiter = mo.ui.slider(5, 60, value=20, step=5, label="Optimizer budget")
    run_fit = mo.ui.run_button(label="Run optimizer")
    mo.hstack([fit_method, fit_sigma_init, fit_maxiter, run_fit], wrap=True)
    return fit_maxiter, fit_method, fit_sigma_init, run_fit


@app.cell
def _(
    fit_maxiter,
    fit_method,
    fit_sigma_init,
    make_opt_population,
    mo,
    opt_budget,
    opt_grid,
    opt_sigma_limits,
    optimize_tuning_width,
    run_fit,
    sweep_duration,
    sweep_seed,
    sweep_stimuli,
    sweep_trials,
):
    mo.stop(
        not run_fit.value,
        mo.md("_Click **Run optimizer** to search for the best feasible σ._"),
    )

    fit_pop = make_opt_population()
    with mo.status.spinner(
        title=f"Optimizing σ with {fit_method.value}",
        subtitle="each step encodes and decodes a fresh set of trials",
    ):
        fit_result, fit_trace = optimize_tuning_width(
            fit_pop,
            opt_grid,
            rate_budget=float(opt_budget.value),
            sigma_init=float(fit_sigma_init.value),
            sigma_bounds=opt_sigma_limits,
            method=str(fit_method.value),
            maxiter=int(fit_maxiter.value),
            eval_kwargs=dict(
                n_test_stimuli=int(sweep_stimuli.value),
                n_trials_per_stim=int(sweep_trials.value),
                trial_duration=float(sweep_duration.value),
                random_seed=int(sweep_seed.value),
            ),
        )
    return fit_pop, fit_result, fit_trace


@app.cell
def _(fit_result, fit_trace, go, make_subplots, mo, np):
    trace_sigmas = np.array([r["sigma"] for r in fit_trace])
    trace_rmse = np.array([r["rmse"] for r in fit_trace])
    trace_steps = np.arange(len(fit_trace))
    trace_running_best = np.minimum.accumulate(trace_rmse)

    fig_fit = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Widths visited", "Convergence"),
        horizontal_spacing=0.11,
    )
    fig_fit.add_trace(
        go.Scatter(
            x=trace_sigmas,
            y=trace_rmse,
            mode="markers",
            name="Evaluations",
            marker=dict(
                size=11,
                color=trace_steps,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="step", x=0.44, thickness=12),
            ),
            hovertemplate="step %{marker.color}<br>σ=%{x:.1f} µs<br>RMSE=%{y:.2f} µs<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_fit.add_trace(
        go.Scatter(
            x=trace_steps,
            y=trace_rmse,
            mode="lines+markers",
            name="RMSE",
            line=dict(color="#4C78A8", width=2),
            hovertemplate="step %{x}<br>RMSE=%{y:.2f} µs<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_fit.add_trace(
        go.Scatter(
            x=trace_steps,
            y=trace_running_best,
            mode="lines",
            name="Running best",
            line=dict(color="#B279A2", width=2, dash="dash"),
            hovertemplate="step %{x}<br>best=%{y:.2f} µs<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_fit.update_xaxes(title_text="σ (µs)", row=1, col=1)
    fig_fit.update_xaxes(title_text="Evaluation", row=1, col=2)
    fig_fit.update_yaxes(title_text="Decoding RMSE (µs)", rangemode="tozero")
    fig_fit.update_layout(
        height=400,
        margin=dict(t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
    )

    fit_message = f"{fit_result.message} ({len(fit_trace)} evaluations)"
    mo.vstack([fig_fit, mo.md(f"_{fit_message}_")])
    return


@app.cell
def _(fit_pop, fit_result, fit_trace, mo, np, opt_budget, opt_grid):
    fit_sigma = float(fit_result.x[0])
    fit_best = min(fit_trace, key=lambda r: r["rmse"])
    fit_start = fit_trace[0]
    fit_rates = fit_pop.compute_rates(opt_grid)
    fit_total_rate = float(np.sum(np.mean(fit_rates, axis=1)))
    fit_improvement = 100.0 * (1.0 - fit_best["rmse"] / fit_start["rmse"])

    fit_stats = mo.hstack(
        [
            mo.stat(f"{fit_sigma:.1f} µs", label="Optimized σ", bordered=True),
            mo.stat(f"{fit_best['rmse']:.2f} µs", label="Best RMSE", bordered=True),
            mo.stat(
                f"{fit_improvement:.1f}%",
                label="RMSE improvement",
                caption=f"from {fit_start['rmse']:.2f} µs at σ={fit_start['sigma']:.1f} µs",
                bordered=True,
            ),
            mo.stat(
                f"{fit_total_rate:.1f} Hz",
                label="Total rate",
                caption=f"budget {float(opt_budget.value):.0f} Hz",
                bordered=True,
            ),
        ],
        wrap=True,
    )

    fit_table = mo.ui.table(
        [
            {
                "neuron": idx,
                "preferred ITD (µs)": round(float(fit_pop.means[idx]), 1),
                "σ (µs)": round(float(fit_pop.sigmas[idx]), 1),
                "avg rate (Hz)": round(float(np.mean(fit_rates[idx, :])), 2),
            }
            for idx in range(fit_pop.n_neurons)
        ],
        selection=None,
        label="Optimized population",
    )
    mo.vstack([fit_stats, fit_table])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Takeaways

    - An optimal tuning width exists: too narrow leaves coverage gaps, too broad
      wastes spikes on uninformative neurons.
    - Under area-conserving normalization the rate budget is nearly flat in
      $\sigma$, so it constrains population size and baseline rate more than
      width. Shrink the budget or raise the neuron count to make it bind.
    - The objective is a noisy black box (spikes are resampled per evaluation),
      so derivative-based methods can stall; a coarse sweep plus a local
      optimizer is more reliable than the optimizer alone.
    """)
    return


if __name__ == "__main__":
    app.run()
