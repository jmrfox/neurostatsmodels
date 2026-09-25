"""Project 10: recurrent excitation, inhibition, and network regimes."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from brian2 import (
        BrianLogger,
        Hz,
        Network,
        NeuronGroup,
        PoissonInput,
        SpikeMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
        prefs,
        second,
        seed as brian_seed,
        start_scope,
    )
    from plotly.subplots import make_subplots
    from scipy.signal import welch

    BrianLogger.log_level_error()
    prefs.codegen.target = "numpy"
    np.set_printoptions(precision=4, suppress=True, linewidth=1000)
    return (
        BrianLogger,
        Hz,
        Network,
        NeuronGroup,
        PoissonInput,
        SpikeMonitor,
        Synapses,
        brian_seed,
        defaultclock,
        go,
        make_subplots,
        mo,
        ms,
        mV,
        np,
        prefs,
        second,
        start_scope,
        welch,
    )


@app.cell
def _(
    Hz,
    Network,
    NeuronGroup,
    PoissonInput,
    SpikeMonitor,
    Synapses,
    brian_seed,
    defaultclock,
    ms,
    mV,
    np,
    second,
    start_scope,
    welch,
):
    # Fixed LIF and input constants. Sliders cover the couplings the project varies.
    TAU_M_MS = 20.0
    EL_MV = -70.0
    VTH_MV = -50.0
    VRESET_MV = -65.0
    TREF_MS = 2.0
    DELAY_MS = 1.0
    W_EXT_MV = 1.0
    N_EXT = 100
    BIN_MS = 5.0

    # Round cuts for the demo, not a fitted classifier.
    QUIET_HZ = 1.0
    RUNAWAY_HZ = 80.0
    SUPPRESS_FRACTION = 0.5
    OSC_RATIO = 8.0
    SYNC_MIN = 2.0

    PRESETS = {
        "asynchronous": dict(w_exc=1.2, w_inh=6.0, nu_ext=40.0, p_connect=0.10, tau_syn=5.0, sigma=4.0),
        "quiescent": dict(w_exc=0.4, w_inh=2.0, nu_ext=10.0, p_connect=0.10, tau_syn=5.0, sigma=1.5),
        "synchronized": dict(w_exc=6.0, w_inh=32.0, nu_ext=34.0, p_connect=0.20, tau_syn=5.0, sigma=3.0),
        "oscillatory": dict(w_exc=2.0, w_inh=8.0, nu_ext=45.0, p_connect=0.18, tau_syn=5.0, sigma=0.3),
        "runaway": dict(w_exc=8.0, w_inh=1.5, nu_ext=40.0, p_connect=0.12, tau_syn=5.0, sigma=1.5),
        "inhibition-suppressed": dict(w_exc=0.5, w_inh=30.0, nu_ext=55.0, p_connect=0.15, tau_syn=5.0, sigma=2.0),
    }

    REGIME_ORDER = [
        "quiescent",
        "asynchronous",
        "synchronized",
        "oscillatory",
        "runaway",
        "inhibition-dominated suppression",
    ]
    REGIME_SHORT = {
        "quiescent": "quiet",
        "asynchronous": "async",
        "synchronized": "sync",
        "oscillatory": "osc",
        "runaway": "run",
        "inhibition-dominated suppression": "inh",
    }
    REGIME_COLOR = {
        "quiescent": "#bdbdbd",
        "asynchronous": "#4C78A8",
        "synchronized": "#F58518",
        "oscillatory": "#54A24B",
        "runaway": "#E45756",
        "inhibition-dominated suppression": "#B279A2",
    }

    def simulate_ei_network(
        n,
        duration_ms,
        w_exc_mV,
        w_inh_mV,
        nu_ext_hz,
        p_connect,
        tau_syn_ms,
        sigma_mV,
        seed_value,
        dt_ms=0.1,
    ):
        """Current-based LIF E/I network. Returns spike times (s) and neuron indices.

        τ_m dv/dt = E_L - v + I_syn + σ √τ_m ξ
        τ_syn dI_syn/dt = -I_syn
        Excitatory spikes add +w_exc; inhibitory spikes add -w_inh, after a fixed delay.
        """
        start_scope()
        defaultclock.dt = dt_ms * ms
        brian_seed(int(seed_value))

        n = int(n)
        n_e = int(round(0.8 * n))
        tau_m = TAU_M_MS * ms
        tau_syn = float(tau_syn_ms) * ms
        EL = EL_MV * mV
        Vth = VTH_MV * mV
        Vreset = VRESET_MV * mV
        sigma = float(sigma_mV) * mV
        w_exc = float(w_exc_mV) * mV
        w_inh = float(w_inh_mV) * mV
        namespace = dict(tau_m=tau_m, tau_syn=tau_syn, EL=EL, sigma=sigma, w_exc=w_exc, w_inh=w_inh, Vth=Vth, Vreset=Vreset)
        eqs = """
        dv/dt = (EL - v + I_syn) / tau_m + sigma * xi / sqrt(tau_m) : volt (unless refractory)
        dI_syn/dt = -I_syn / tau_syn : volt
        """
        group = NeuronGroup(
            n,
            eqs,
            threshold="v > Vth",
            reset="v = Vreset",
            refractory=TREF_MS * ms,
            method="euler",
            namespace=namespace,
        )
        group.v = EL
        syn_e = Synapses(group[:n_e], group, on_pre="I_syn_post += w_exc", namespace=namespace)
        syn_i = Synapses(group[n_e:], group, on_pre="I_syn_post -= w_inh", namespace=namespace)
        syn_e.connect(p=float(p_connect), condition="j != i")
        syn_i.connect(p=float(p_connect), condition="j != i + n_e")
        syn_e.delay = DELAY_MS * ms
        syn_i.delay = DELAY_MS * ms
        external = PoissonInput(group, "I_syn", N_EXT, float(nu_ext_hz) * Hz, weight=W_EXT_MV * mV)
        spikes = SpikeMonitor(group)
        Network(group, syn_e, syn_i, external, spikes).run(float(duration_ms) * ms, report=None)
        return np.asarray(spikes.t / second), np.asarray(spikes.i, dtype=int), n_e, n

    def _bin_counts(times, t0, t1, bin_s):
        edges = np.arange(t0, t1 + bin_s * 0.5, bin_s)
        if edges.size < 2:
            edges = np.array([t0, t1])
        counts, edges = np.histogram(times, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts

    def summarize_spikes(spike_t, spike_i, n_e, n, duration_s, transient_s):
        """Rates, synchrony, ISIs, autocorrelation, and spectrum after the transient."""
        t0 = float(transient_s)
        t1 = float(duration_s)
        keep = spike_t >= t0
        spike_t = spike_t[keep]
        spike_i = spike_i[keep]
        analysis_s = max(t1 - t0, 1e-9)
        bin_s = BIN_MS / 1000.0
        centers, counts = _bin_counts(spike_t, t0, t1, bin_s)
        exc_times = spike_t[spike_i < n_e]
        inh_times = spike_t[spike_i >= n_e]
        _, counts_e = _bin_counts(exc_times, t0, t1, bin_s)
        _, counts_i = _bin_counts(inh_times, t0, t1, bin_s)
        n_i = n - n_e
        pop_hz = counts / (n * bin_s)
        exc_hz = counts_e / (max(n_e, 1) * bin_s)
        inh_hz = counts_i / (max(n_i, 1) * bin_s)
        mean_rate = float(spike_t.size / (n * analysis_s))
        mu = float(counts.mean()) if counts.size else 0.0
        if mu > 0 and counts.size > 1:
            sync = float(counts.std(ddof=1) / np.sqrt(mu))
            rate_cv = float(counts.std(ddof=1) / mu)
        else:
            sync = 0.0
            rate_cv = 0.0

        isi_ms = []
        cvs = []
        neuron_hz = np.zeros(n)
        for neuron in range(n):
            st = np.sort(spike_t[spike_i == neuron])
            neuron_hz[neuron] = st.size / analysis_s
            if st.size < 2:
                continue
            isi = np.diff(st)
            isi_ms.append(isi * 1000.0)
            if st.size >= 4 and isi.mean() > 0:
                cvs.append(float(isi.std(ddof=1) / isi.mean()))
        isi_ms = np.concatenate(isi_ms) if isi_ms else np.array([])
        isi_cv = float(np.mean(cvs)) if cvs else np.nan

        if counts.size >= 16 and np.any(counts):
            nperseg = min(counts.size, max(16, int(round(0.25 / bin_s))))
            freqs, spec = welch(counts - counts.mean(), fs=1.0 / bin_s, nperseg=nperseg)
            band = (freqs >= 15.0) & (freqs <= 90.0)
            if np.any(band) and np.max(spec[band]) > 0:
                band_spec = spec[band]
                peak_hz = float(freqs[band][int(np.argmax(band_spec))])
                med = float(np.median(band_spec))
                peak_ratio = float(band_spec.max() / med) if med > 0 else np.inf
            else:
                peak_hz, peak_ratio = np.nan, 0.0
        else:
            freqs = np.array([0.0])
            spec = np.array([0.0])
            peak_hz, peak_ratio = np.nan, 0.0

        x = counts - counts.mean()
        ac = np.correlate(x, x, mode="full")
        ac = ac[ac.size // 2 :]
        if ac.size and ac[0] != 0:
            ac = ac / ac[0]
        lags_ms = np.arange(ac.size) * bin_s * 1000.0
        return dict(
            rate=mean_rate,
            sync=sync,
            rate_cv=rate_cv,
            isi_cv=isi_cv,
            peak_hz=peak_hz,
            peak_ratio=peak_ratio,
            centers=centers,
            pop_hz=pop_hz,
            exc_hz=exc_hz,
            inh_hz=inh_hz,
            isi_ms=isi_ms,
            neuron_hz=neuron_hz,
            lags_ms=lags_ms,
            ac=ac,
            freqs=freqs,
            spec=spec,
            spike_t=spike_t,
            spike_i=spike_i,
            n_e=n_e,
            n=n,
        )

    def classify_regime(summary, uncoupled_rate):
        """Apply the notebook's fixed cuts. First match wins."""
        rate = summary["rate"]
        sync = summary["sync"]
        ratio = summary["peak_ratio"]
        peak = summary["peak_hz"]
        if rate < QUIET_HZ:
            label = "quiescent"
        elif rate > RUNAWAY_HZ:
            label = "runaway"
        elif uncoupled_rate >= 5.0 and rate < SUPPRESS_FRACTION * uncoupled_rate:
            label = "inhibition-dominated suppression"
        elif ratio >= OSC_RATIO and peak == peak and 15.0 <= peak <= 90.0:
            label = "oscillatory"
        elif sync >= SYNC_MIN:
            label = "synchronized"
        else:
            label = "asynchronous"
        return label

    def transient_for(duration_ms):
        return float(min(100.0, 0.25 * float(duration_ms)))

    return (
        BIN_MS,
        DELAY_MS,
        N_EXT,
        OSC_RATIO,
        PRESETS,
        QUIET_HZ,
        REGIME_COLOR,
        REGIME_ORDER,
        REGIME_SHORT,
        RUNAWAY_HZ,
        SUPPRESS_FRACTION,
        SYNC_MIN,
        TAU_M_MS,
        W_EXT_MV,
        classify_regime,
        simulate_ei_network,
        summarize_spikes,
        transient_for,
    )


@app.cell(hide_code=True)
def _(DELAY_MS, N_EXT, TAU_M_MS, W_EXT_MV, mo):
    mo.md(rf"""
    # Recurrent excitation, inhibition, and network regimes

    Project 10. A modest random recurrent network of current-based leaky
    integrate-and-fire neurons, half the work of seeing a regime in a raster and
    half in the population signals (rate, synchrony, spectrum).

    ## Model

    $N_E = 0.8 N$ excitatory cells and $N_I = 0.2 N$ inhibitory cells.
    Membrane and synapses:

    $$
    \tau_m \dot v = E_L - v + I_{{\mathrm{{syn}}}} + \sigma \sqrt{{\tau_m}}\,\xi
    $$

    $$
    \tau_{{\mathrm{{syn}}}} \dot I_{{\mathrm{{syn}}}} = -I_{{\mathrm{{syn}}}}
    $$

    A spike resets $v$ to $V_{{\mathrm{{reset}}}}$ and starts a {2:g} ms refractory period.
    After a fixed axonal delay of {DELAY_MS:g} ms, an excitatory spike adds $+w_{{\mathrm{{exc}}}}$
    to $I_{{\mathrm{{syn}}}}$ and an inhibitory spike adds $-w_{{\mathrm{{inh}}}}$.
    Each cell also receives {N_EXT:g} independent Poisson inputs at rate $\nu_{{\mathrm{{ext}}}}$
    with weight {W_EXT_MV:g} mV, so the sparsity slider changes recurrence without
    rescaling the external current.

    Fixed: $\tau_m = {TAU_M_MS:g}$ ms, $E_L = -70$ mV, $V_{{\mathrm{{th}}}} = -50$ mV,
    $V_{{\mathrm{{reset}}}} = -65$ mV. $\xi$ is unit white noise.
    """)
    return


@app.cell(hide_code=True)
def _(OSC_RATIO, QUIET_HZ, RUNAWAY_HZ, SUPPRESS_FRACTION, SYNC_MIN, mo):
    mo.md(rf"""
    ## One network

    Load a preset to set the six sliders, then change any of them.
    **Run network** simulates this connectivity and the same drive with
    $w_{{\mathrm{{exc}}}} = w_{{\mathrm{{inh}}}} = 0$.

    The label uses these fixed cuts, in order:

    - quiescent: rate $< {QUIET_HZ:g}$ Hz
    - runaway: rate $> {RUNAWAY_HZ:g}$ Hz
    - inhibition-dominated suppression: rate $< {SUPPRESS_FRACTION:g}$ times the uncoupled rate
    - oscillatory: spectral peak / median $\geq {OSC_RATIO:g}$ between 15 and 90 Hz
    - synchronized: synchrony $S \geq {SYNC_MIN:g}$ without that peak
    - otherwise asynchronous

    $S = \mathrm{{std}}(K) / \sqrt{{\mathrm{{mean}}(K)}}$ for population spike counts $K$ in 5 ms bins.
    Irregular independent spiking sits near $S \approx 1$. Presets are rough starting
    points from short runs at $N = 400$; a different $N$ or seed can cross a cut.
    """)
    return


@app.cell
def _(PRESETS, mo):
    preset = mo.ui.dropdown(options=list(PRESETS.keys()), value="asynchronous", label="Load preset")
    preset
    return (preset,)


@app.cell
def _(PRESETS, mo, preset):
    chosen = PRESETS[preset.value]
    w_exc = mo.ui.slider(0.0, 10.0, value=chosen["w_exc"], step=0.1, label="w_exc (mV)")
    w_inh = mo.ui.slider(0.0, 40.0, value=chosen["w_inh"], step=0.5, label="w_inh (mV)")
    nu_ext = mo.ui.slider(0.0, 80.0, value=chosen["nu_ext"], step=1.0, label="ν_ext (Hz)")
    p_connect = mo.ui.slider(0.02, 0.40, value=chosen["p_connect"], step=0.01, label="Sparsity p")
    tau_syn = mo.ui.slider(2.0, 20.0, value=chosen["tau_syn"], step=0.5, label="τ_syn (ms)")
    sigma = mo.ui.slider(0.0, 8.0, value=chosen["sigma"], step=0.1, label="Noise σ (mV)")
    mo.vstack(
        [
            mo.hstack([w_exc, w_inh, nu_ext], wrap=True),
            mo.hstack([p_connect, tau_syn, sigma], wrap=True),
        ]
    )
    return nu_ext, p_connect, sigma, tau_syn, w_exc, w_inh


@app.cell
def _(mo):
    n_neurons = mo.ui.slider(100, 500, value=400, step=50, label="N")
    duration_ms = mo.ui.slider(200, 800, value=400, step=50, label="Duration (ms)")
    sim_seed = mo.ui.number(0, 9999, value=1, step=1, label="Seed")
    run_sim = mo.ui.run_button(label="Run network")
    mo.hstack([n_neurons, duration_ms, sim_seed, run_sim], wrap=True)
    return duration_ms, n_neurons, run_sim, sim_seed


@app.cell
def _(
    classify_regime,
    duration_ms,
    mo,
    n_neurons,
    nu_ext,
    p_connect,
    run_sim,
    sigma,
    sim_seed,
    simulate_ei_network,
    summarize_spikes,
    tau_syn,
    transient_for,
    w_exc,
    w_inh,
):
    mo.stop(not run_sim.value, mo.md("_Click **Run network** to simulate._"))
    duration = float(duration_ms.value)
    transient = transient_for(duration)
    common = dict(
        n=int(n_neurons.value),
        duration_ms=duration,
        nu_ext_hz=float(nu_ext.value),
        p_connect=float(p_connect.value),
        tau_syn_ms=float(tau_syn.value),
        sigma_mV=float(sigma.value),
        seed_value=int(sim_seed.value),
    )
    spike_t, spike_i, n_e, n = simulate_ei_network(
        w_exc_mV=float(w_exc.value),
        w_inh_mV=float(w_inh.value),
        **common,
    )
    summary = summarize_spikes(spike_t, spike_i, n_e, n, duration / 1000.0, transient / 1000.0)
    bare_t, bare_i, bare_ne, bare_n = simulate_ei_network(w_exc_mV=0.0, w_inh_mV=0.0, **common)
    uncoupled = summarize_spikes(bare_t, bare_i, bare_ne, bare_n, duration / 1000.0, transient / 1000.0)
    label = classify_regime(summary, uncoupled["rate"])
    return label, summary, transient, uncoupled


@app.cell
def _(go, label, mo, np, summary, transient, uncoupled):
    _exc = summary["spike_i"] < summary["n_e"]
    fig_raster = go.Figure()
    fig_raster.add_trace(
        go.Scattergl(
            x=summary["spike_t"][_exc],
            y=summary["spike_i"][_exc],
            mode="markers",
            name="Excitatory",
            marker=dict(size=3, color="#4C78A8"),
        )
    )
    fig_raster.add_trace(
        go.Scattergl(
            x=summary["spike_t"][~_exc],
            y=summary["spike_i"][~_exc],
            mode="markers",
            name="Inhibitory",
            marker=dict(size=3, color="#E45756"),
        )
    )
    fig_raster.update_layout(
        height=420,
        title=f"Raster after {transient:.0f} ms transient — {label}",
        xaxis_title="Time (s)",
        yaxis_title="Neuron",
        margin=dict(t=60, b=40),
    )

    fig_rate = go.Figure()
    fig_rate.add_trace(go.Scatter(x=summary["centers"], y=summary["pop_hz"], name="All", line=dict(color="#333333")))
    fig_rate.add_trace(go.Scatter(x=summary["centers"], y=summary["exc_hz"], name="E", line=dict(color="#4C78A8")))
    fig_rate.add_trace(go.Scatter(x=summary["centers"], y=summary["inh_hz"], name="I", line=dict(color="#E45756")))
    fig_rate.update_layout(
        height=320,
        title="Population rate",
        xaxis_title="Time (s)",
        yaxis_title="Rate (Hz)",
        margin=dict(t=50, b=40),
    )

    _isi = summary["isi_ms"]
    _isi_show = _isi[_isi <= 200] if _isi.size else _isi
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("ISI distribution", "Firing-rate distribution"))
    fig_dist.add_trace(go.Histogram(x=_isi_show, nbinsx=40, marker_color="#4C78A8", name="ISI"), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=summary["neuron_hz"], nbinsx=30, marker_color="#54A24B", name="Rate"), row=1, col=2)
    fig_dist.update_xaxes(title_text="ISI (ms)", row=1, col=1)
    fig_dist.update_xaxes(title_text="Rate (Hz)", row=1, col=2)
    fig_dist.update_yaxes(title_text="Count", row=1, col=1)
    fig_dist.update_layout(height=320, showlegend=False, margin=dict(t=50, b=40))

    _max_lag = min(summary["lags_ms"].size, int(150 / 5) + 1)
    fig_dyn = make_subplots(rows=1, cols=2, subplot_titles=("Population-rate autocorrelation", "Power spectrum"))
    fig_dyn.add_trace(
        go.Scatter(x=summary["lags_ms"][:_max_lag], y=summary["ac"][:_max_lag], line=dict(color="#4C78A8"), name="AC"),
        row=1,
        col=1,
    )
    fig_dyn.add_trace(
        go.Scatter(x=summary["freqs"], y=summary["spec"], line=dict(color="#F58518"), name="Spectrum"),
        row=1,
        col=2,
    )
    fig_dyn.update_xaxes(title_text="Lag (ms)", row=1, col=1)
    fig_dyn.update_xaxes(title_text="Frequency (Hz)", range=[0, 120], row=1, col=2)
    fig_dyn.update_yaxes(title_text="Autocorr", row=1, col=1)
    fig_dyn.update_yaxes(title_text="Power", row=1, col=2)
    fig_dyn.update_layout(height=320, showlegend=False, margin=dict(t=50, b=40))

    _isi_cv = summary["isi_cv"]
    _isi_txt = "n/a" if _isi_cv != _isi_cv else f"{_isi_cv:.2f}"
    _peak = summary["peak_hz"]
    _peak_txt = "n/a" if _peak != _peak else f"{_peak:.0f} Hz"
    _hints = {
        "quiescent": "The raster is almost empty, and the rate trace stays at rest. There is no hidden population rhythm underneath.",
        "asynchronous": "The raster looks like unstructured noise. A flat population rate and S near 1 mean those spikes are not shared volleys.",
        "synchronized": "Rows look irregular on their own. The population rate still jumps when many cells fire together, which is what pushes S up.",
        "oscillatory": "Single rows look noisy. The spectrum of the population rate shows a repeating rhythm that is hard to pick out one neuron at a time.",
        "runaway": "The raster fills in. The rate says how close the population is to firing as fast as the refractory period allows.",
        "inhibition-dominated suppression": "Spikes are still there, so the raster does not look shut off. The comparison is the uncoupled rate at the same external drive.",
    }
    note = mo.md(
        f"**{label}.** Mean rate {summary['rate']:.1f} Hz "
        f"(uncoupled {uncoupled['rate']:.1f} Hz). "
        f"Synchrony $S$ = {summary['sync']:.2f}, population-rate CV = {summary['rate_cv']:.2f}, "
        f"mean ISI CV = {_isi_txt}. "
        f"Spectral peak {_peak_txt}, peak/median = {summary['peak_ratio']:.1f}. "
        f"{_hints[label]}"
    )
    mo.vstack([fig_raster, note, fig_rate, fig_dist, fig_dyn])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coupling phase diagram

    Sweep a $6 \times 6$ grid of $w_{\mathrm{exc}}$ and $w_{\mathrm{inh}}$ at the
    drive, sparsity, $\tau_{\mathrm{syn}}$, noise, and seed currently set above.
    Each point uses $N = 200$ and 250 ms, then the same cuts as the single-network
    label. One extra run with both weights at zero supplies the uncoupled rate.
    This is a coarse sketch: short traces make the spectral cut noisy.
    """)
    return


@app.cell
def _(mo):
    run_phase = mo.ui.run_button(label="Run phase diagram")
    run_phase
    return (run_phase,)


@app.cell
def _(
    REGIME_COLOR,
    REGIME_ORDER,
    REGIME_SHORT,
    classify_regime,
    go,
    make_subplots,
    mo,
    np,
    nu_ext,
    p_connect,
    run_phase,
    sigma,
    sim_seed,
    simulate_ei_network,
    summarize_spikes,
    tau_syn,
):
    mo.stop(not run_phase.value, mo.md("_Click **Run phase diagram** to sweep coupling._"))
    _n_phase = 200
    _duration_ms = 250.0
    _transient_s = 0.05
    _w_exc_axis = np.linspace(0.0, 8.0, 6)
    _w_inh_axis = np.linspace(0.0, 32.0, 6)
    _common = dict(
        n=_n_phase,
        duration_ms=_duration_ms,
        nu_ext_hz=float(nu_ext.value),
        p_connect=float(p_connect.value),
        tau_syn_ms=float(tau_syn.value),
        sigma_mV=float(sigma.value),
        seed_value=int(sim_seed.value),
    )
    _bare_t, _bare_i, _bare_ne, _bare_n = simulate_ei_network(w_exc_mV=0.0, w_inh_mV=0.0, **_common)
    _uncoupled_rate = summarize_spikes(
        _bare_t, _bare_i, _bare_ne, _bare_n, _duration_ms / 1000.0, _transient_s
    )["rate"]
    _jobs = [(i, j, we, wi) for i, wi in enumerate(_w_inh_axis) for j, we in enumerate(_w_exc_axis)]
    _labels = np.empty((_w_inh_axis.size, _w_exc_axis.size), dtype=object)
    _rates = np.zeros_like(_labels, dtype=float)
    for _i, _j, _we, _wi in mo.status.progress_bar(_jobs, title="Coupling grid"):
        _t, _idx, _ne, _n = simulate_ei_network(w_exc_mV=float(_we), w_inh_mV=float(_wi), **_common)
        _point = summarize_spikes(_t, _idx, _ne, _n, _duration_ms / 1000.0, _transient_s)
        _labels[_i, _j] = classify_regime(_point, _uncoupled_rate)
        _rates[_i, _j] = _point["rate"]

    _codes = np.vectorize(REGIME_ORDER.index)(_labels)
    _colors = [REGIME_COLOR[name] for name in REGIME_ORDER]
    _n_colors = len(_colors)
    _scale = []
    for _k, _color in enumerate(_colors):
        _scale.append([_k / _n_colors, _color])
        _scale.append([(_k + 1) / _n_colors, _color])
    fig_phase = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Regime label", "Mean rate (Hz)"),
        horizontal_spacing=0.12,
    )
    fig_phase.add_trace(
        go.Heatmap(
            z=_codes,
            x=_w_exc_axis,
            y=_w_inh_axis,
            text=np.vectorize(REGIME_SHORT.get)(_labels),
            texttemplate="%{text}",
            customdata=_labels,
            colorscale=_scale,
            zmin=-0.5,
            zmax=len(REGIME_ORDER) - 0.5,
            showscale=False,
            hovertemplate="w_exc=%{x:.1f}<br>w_inh=%{y:.1f}<br>%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_phase.add_trace(
        go.Heatmap(
            z=_rates,
            x=_w_exc_axis,
            y=_w_inh_axis,
            colorscale="Viridis",
            colorbar=dict(title="Hz", x=1.0),
            hovertemplate="w_exc=%{x:.1f}<br>w_inh=%{y:.1f}<br>%{z:.1f} Hz<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_phase.update_xaxes(title_text="w_exc (mV)")
    fig_phase.update_yaxes(title_text="w_inh (mV)")
    fig_phase.update_layout(height=480, margin=dict(t=60, b=40))
    _phase_note = mo.md(
        f"Uncoupled rate at this drive: **{_uncoupled_rate:.1f} Hz** "
        f"($N = {_n_phase}$, {_duration_ms:.0f} ms). "
        "Labels are the same ordered cuts as above."
    )
    mo.vstack([fig_phase, _phase_note])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Takeaways

    - Recurrent excitation and inhibition do not just scale the firing rate.
      The same external drive can be quiet, asynchronous, synchronized, rhythmic,
      suppressed, or running away, depending on the two weights.
    - A raster shows who spiked. The population rate, $S$, and the spectrum say
      whether those spikes are shared, rhythmic, or only locally irregular.
    - The cuts are deliberate round numbers so the diagram stays readable.
      They are not a claim that real cortex falls into six sharp boxes.
    """)
    return


if __name__ == "__main__":
    app.run()
