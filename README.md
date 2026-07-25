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

Edit the Project 1 first pass:

```bash
uv run marimo edit marimo/p01_tuning_fi_coding/itd_information.py
```

Optional intro tutorial:

```bash
uv run marimo tutorial intro
```
