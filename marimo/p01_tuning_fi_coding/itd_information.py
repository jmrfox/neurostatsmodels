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

    np.set_printoptions(precision=4, suppress=True, linewidth=1000)
    return (
        CopyToClipboard,
        GaussianTunedPopulation,
        Slider2D,
        go,
        make_subplots,
        mo,
        np,
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

    Plots use **Plotly** (hover, zoom, toggle traces). Constrained $\sigma$
    optimization under a rate budget still lives in
    `jupyter/p01_tuning_fi_coding/`.
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
    ### Next steps

    The jupyter original also sweeps $\sigma$ and runs a constrained COBYLA
    optimizer under a population rate budget. Port that interactively once the
    objective evaluation is cheaper (or cached).
    """)
    return


if __name__ == "__main__":
    app.run()
