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
    duration = T - 2 * delay
    t = np.arange(0, T + dt, dt)
    x = np.zeros(len(t))
    x[(t > delay) & (t < delay + duration)] = amplitude
    return x
