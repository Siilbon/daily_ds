# daily_ds

A scratchpad for small data science exercises. Each file in `src/daily_ds/`
is one self-contained exercise, written as `#%%` cells so it runs in the
VS Code Interactive Window (Shift+Enter).

## Setup

```sh
uv sync
```

Then select the `.venv` interpreter in VS Code.

## Adding an exercise

Create `src/daily_ds/<topic>.py`, start it with `#%%`, and keep it standalone:
generate or load its own data, no imports from other exercises.

Add a package with `uv add <name>`.

Local datasets go in `data/` (git-ignored).

## Exercises

| File | Topic |
|---|---|
| `feature_scaling.py` | When scaling matters (KNN vs random forest, robust scaling with spikes) |
| `partial_autocorrelation.py` | ACF/PACF on an AR(2) process with trend |
| `sklean_transformers.py` | Writing a custom sklearn transformer inside a pipeline |
| `tssplit.py` | TimeSeriesSplit vs KFold leakage on time series |
