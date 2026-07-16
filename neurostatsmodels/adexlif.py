"""Adaptive exponential leaky integrate-and-fire (AdEx) neuron model.

Implements the Brette–Gerstner AdEx equations with optional spike jitter and
probabilistic spiking. Useful for exploring how adaptation and exponential
spike initiation shape responses to injected current.
"""

from typing import Sequence
import numpy as np


def adexlif_simulation(
    current_input: Sequence[float],  # input current timeseries in pA
    dt: float,  # time step in ms
    Vrest: float,  # resting potential in mV
    Vreset: float,  # reset potential in mV
    VT: float,  # threshold potential in mV
    Vthres: float,  # spike threshold in mV
    EL: float,  # leak reversal potential in mV
    Ew: float,  # reset potential of the adaptation current in mV
    Tref: float,  # refractory period in ms
    tRC: float,  # time constant of the leak in ms
    tau_w: float,  # time constant of the adaptation current in ms
    R: float,  # resistance in MOhm
    Del: float,  # slope factor in mV
    a: float,  # subthreshold adaptation conductance in nS
    b: float,  # spike-triggered adaptation current in pA
    jitter_range: float = 0.0,  # jitter range for spike times in ms
    spike_probability: float = 1.0,  # probability of spike occurrence
    **kwargs,  # additional parameters (not used in this function, but allows for flexibility in future use
):
    """Simulate a single AdEx neuron driven by a current injection.

    Numerically integrates the adaptive exponential LIF dynamics:

    - Membrane voltage ``V`` evolves with leak, exponential spike initiation,
      adaptation current ``w``, and injected current.
    - Adaptation ``w`` tracks ``V`` with time constant ``tau_w`` (parameter ``a``)
      and jumps by ``b`` on each spike.
    - When ``V`` crosses ``Vthres``, the cell resets to ``Vreset``, enters
      refractory period ``Tref``, and may record a spike (subject to
      ``spike_probability`` and optional temporal jitter).

    Intent: provide a transparent single-cell simulator for studying how
    adaptation (``a``, ``b``, ``tau_w``) and exponential onset (``Del``, ``VT``)
    interact with current waveforms, without requiring a full network library.

    Parameters
    ----------
    current_input : sequence of float
        Injected current over time, in pA. Length sets the simulation duration.
    dt : float
        Integration time step in ms.
    Vrest : float
        Initial membrane potential in mV (also used to seed ``w``).
    Vreset : float
        Post-spike reset potential in mV.
    VT : float
        Soft threshold / rheobase-related potential in the exponential term (mV).
    Vthres : float
        Hard spike-detection threshold in mV; crossing triggers reset.
    EL : float
        Leak reversal potential in mV.
    Ew : float
        Adaptation current reversal / resting level in mV.
    Tref : float
        Absolute refractory period in ms after a spike.
    tRC : float
        Membrane time constant in ms (scales the voltage update).
    tau_w : float
        Adaptation current time constant in ms.
    R : float
        Membrane resistance in MOhm (converts currents to voltages).
    Del : float
        Slope factor of the exponential spike initiation in mV.
    a : float
        Subthreshold adaptation conductance in nS.
    b : float
        Spike-triggered jump in adaptation current in pA.
    jitter_range : float, optional
        Full width of uniform spike-time jitter in ms. Spike times are shifted
        by ``U(-0.5, 0.5) * jitter_range``. Default is 0 (no jitter).
    spike_probability : float, optional
        Probability of recording a spike when threshold is crossed. Values
        below 1 introduce stochastic failures. Default is 1.0.
    **kwargs
        Ignored; accepted so callers can pass unused parameter bags.

    Returns
    -------
    dict
        ``t`` : ndarray
            Time axis in ms.
        ``Vm`` : ndarray
            Membrane voltage trace in mV (spiking events clipped for display).
        ``w`` : ndarray
            Adaptation current trace.
        ``spike_times`` : ndarray
            Recorded spike times in ms (possibly jittered).
        ``spike_count`` : int
            Number of recorded spikes.
    """

    # Numerical integration
    eps = dt / tRC

    # Get population size and number of time steps in input signal
    n_timesteps = len(current_input)

    # Time
    t_domain = np.arange(0, n_timesteps) * dt

    # Intialize
    refractory_timer = 0
    spike_count = 0
    spike_times = []

    V = Vrest

    w = a * (V - Ew)

    Vm = np.ones(n_timesteps) * Vreset
    Vm[0] = V

    w_out = np.zeros(n_timesteps)
    w_out[0] = w

    # Run the simulation
    for n in range(n_timesteps - 1):

        # Membrane potential update

        dV = (EL - V) + Del * np.exp((V - VT) / Del) - R * w + R * current_input[n]

        # Adaptation current
        w += dt / tau_w * (a * (V - Ew) - w)
        w_out[n + 1] = w

        # If out of refractory period, update membrane potential
        if refractory_timer <= 0:
            V += eps * dV
            Vm[n + 1] = V

        # Update adaptation current
        if V > Vthres:
            w += b

        # Randomly select spikes to keep
        spike_success = (V > Vthres) & (np.random.uniform() < spike_probability)

        # Record the spike times with jitter added to the spike time
        if spike_success:
            spike_count += 1
            spike_times.append(t_domain[n] + np.random.uniform(-0.5, 0.5) * jitter_range)

        if V > Vthres:
            # Reset the refactory period counter
            refractory_timer = Tref
            # Reset the membrane potential to the reset value
            V = Vreset

            Vm[n + 1] = -10

        # Decrease the refactory period counter for those in the refractory period
        if refractory_timer > 0:
            # Decrease the refractory period counter
            refractory_timer -= dt

    # END of time loop. Process the data to return

    Vm[Vm > -10] = -10

    spike_times = np.array(spike_times)
    trace = np.array([t_domain, Vm]).T
    output_dict = {
        "t": t_domain,
        "Vm": Vm,
        "w": w_out,
        "spike_times": spike_times,
        "spike_count": spike_count,
    }
    return output_dict


def get_step_current(
    amplitude: float,
    delay: float,
    T: float,
    dt: float,
):
    """Build a rectangular (step) current injection waveform.

    Returns a zero current of total length ``T``, with a constant pulse of
    ``amplitude`` that starts after ``delay`` and ends ``delay`` before ``T``
    (so the on-duration is ``T - 2 * delay``). Useful as a simple drive for
    ``adexlif_simulation`` and related single-cell experiments.

    Parameters
    ----------
    amplitude : float
        Pulse amplitude (same units as the neuron model's current input, e.g. pA).
    delay : float
        Quiet interval before and after the pulse, in the same time units as
        ``T`` and ``dt`` (typically ms).
    T : float
        Total waveform duration (typically ms).
    dt : float
        Sampling interval (typically ms).

    Returns
    -------
    ndarray
        Current values sampled at ``dt``, length ``len(arange(0, T + dt, dt))``.
    """
    duration = T - 2 * delay
    t = np.arange(0, T + dt, dt)
    x = np.zeros(len(t))
    x[(t > delay) & (t < delay + duration)] = amplitude
    return x
