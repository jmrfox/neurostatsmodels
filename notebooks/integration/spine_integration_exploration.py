# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from neurostatsmodels.integration import *

seed = None
n_trials = 100
total_time_ms = 1000
n_epochs = 3
nonlinear = True
per_trial_normalization = 'unit_gaussian' #'divide_by_max'
total_rate_hz = 10

# typical spine
typical_options = IntegratorOptions(
    total_time_ms=total_time_ms,
    n_epochs=n_epochs,
    random_seed=seed,
    n_trials=n_trials,
    n_synapses=1,
    morphology_amplitude_range=(10.0, 10.0),
    morphology_tau_range=(1.0, 1.0),
    nonlinear=nonlinear,
    total_rate_hz=total_rate_hz,
)
typical_sim = Integrator(typical_options)

# complex spine
complex_options = IntegratorOptions(
    total_time_ms=total_time_ms,
    n_epochs=n_epochs,
    random_seed=seed,
    n_trials=n_trials,
    n_synapses=10,
    morphology_amplitude_range=(1.0, 10.0),
    morphology_tau_range=(1.0, 20.0),
    nonlinear=nonlinear,
    active_synapse_fraction='random',
    total_rate_hz=total_rate_hz,
)
complex_sim = Integrator(complex_options)

rng = np.random.default_rng(seed=seed)

print("Running typical spine...")
Y_typical = typical_sim.run_trials(per_trial_normalization=per_trial_normalization)

print("Running complex spine...")
Y_complex = complex_sim.run_trials(per_trial_normalization=per_trial_normalization)

pr_typical, sv_typical = typical_sim.compute_participation_ratio(Y_typical)
pr_complex, sv_complex = complex_sim.compute_participation_ratio(Y_complex)

print("\nParticipation ratio")
print("Typical spine:")
print(f"PR = {pr_typical:0.3}\tPR/n = {pr_typical/n_trials:0.3}")
print("Complex spine:")
print(f"PR = {pr_complex:0.3}\tPR/n = {pr_complex/n_trials:0.3}")

# %%
example_idx = rng.integers(low=0, high=n_trials)
plt.figure(figsize=(8, 4))
plt.plot(typical_sim.time, Y_typical[example_idx], label="Typical")
plt.plot(complex_sim.time, Y_complex[example_idx], label="Complex")
plt.xlabel("Time (ms)")
plt.ylabel("Output voltage")
plt.title(f"Example: trial #{example_idx}")
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(sv_typical, "o-", label="Typical")
plt.plot(sv_complex, "o-", label="Complex")
plt.xlabel("Component")
plt.ylabel("Raw singular value")
plt.title("PCA spectra")
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(sv_typical / sv_typical[0], "o-", label="Typical")
plt.plot(sv_complex / sv_complex[0], "o-", label="Complex")
plt.xlabel("Component")
plt.ylabel("Normalized singular value")
plt.title("Normalized PCA spectra")
plt.legend()
plt.show()

# %%
analyzer = BatchSignalAnalyzer(Y_typical, Y_complex)
analyzer.run_all_print()

# nnanalyzer = NonnegativeBatchSignalAnalyzer(Y_typical, Y_complex)
# nnanalyzer.run_all_print(max_iter=1000)

# %%
from sklearn.decomposition import PCA

def pca_reconstruction_error(V, k):
    """Relative Frobenius error of reconstructing ``V`` with ``k`` PCA components.

    Fits PCA on the trial×time matrix ``V``, projects to ``k`` components, and
    returns ``||V - V_hat|| / ||V||``. Used to compare how many linear modes
    "typical" vs "complex" integrator ensembles need.

    Parameters
    ----------
    V : ndarray, shape (n_trials, n_features)
        Data matrix (e.g. voltage traces across trials).
    k : int
        Number of principal components to retain.

    Returns
    -------
    float
        Relative reconstruction error in ``[0, 1+]`` (0 = perfect).
    """
    pca = PCA(n_components=k)
    Z = pca.fit_transform(V)
    V_hat = pca.inverse_transform(Z)

    error = np.linalg.norm(V - V_hat) / np.linalg.norm(V)
    return error

pcare_typical = []
pcare_complex = []
for k in range(1, 21):
    pcare_typical.append(pca_reconstruction_error(Y_typical, k))
    pcare_complex.append(pca_reconstruction_error(Y_complex, k))
plt.plot(pcare_typical, label='Typical', marker='o')
plt.plot(pcare_complex, label='Complex', marker='s')
plt.legend()
plt.show()

# %%
