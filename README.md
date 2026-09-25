# Computational Neuroscience Models

Space for computational neuroscience models and simulations.

## Setup

```bash
uv sync
```

## Notebooks

| Directory | Role |
|-----------|------|
| `marimo/` | Interactive marimo notebooks (preferred for new work) |
| `jupyter/` | Legacy Jupyter + Jupytext pairs |

Interactivity: marimo `mo.ui` widgets, plus [wigglystuff](https://koaning.github.io/wigglystuff/) via `mo.ui.anywidget(...)`.

Edit Project 1 (ITD / Fisher information):

```bash
uv run marimo edit marimo/p01_tuning_fi_coding/itd_information.py
```

Edit Project 3 (shared variability / latent gain):

```bash
uv run marimo edit marimo/p03_shared_variability/latent_gain.py
```

Edit Project 10 (E/I network regimes):

```bash
uv run marimo edit marimo/p10_ei_regimes/ei_regimes.py
```

Optional intro tutorial:

```bash
uv run marimo tutorial intro
```
