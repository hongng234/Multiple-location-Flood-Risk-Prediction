# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is
Multi-location **next-day flood-risk forecasting** for 5 Vietnamese locations. Daily rainfall/temperature (Open-Meteo Archive) + river discharge (GloFAS via Open-Meteo Flood API) → per-location binary label "discharge at t+horizon > train-period P90" → benchmark of baselines and sklearn classifiers.

## Commands (Windows venv; `make` targets in parentheses)
- Install: `.venv/Scripts/python -m pip install -r requirements.txt` (`make install`)
- Train + benchmark: `.venv/Scripts/python main.py` (`make run`); `--refresh` re-downloads (`make refresh`)
- Live forecast: `.venv/Scripts/python predict.py` (`make predict`) — needs a trained `outputs/models/best_model.joblib`
- Tests (offline, ~15 s): `.venv/Scripts/python -m pytest` (`make test`)
- Offline full run: set `data.source: "synthetic"` in a copy of `config.yaml`, then `main.py --config <copy>`

## Architecture
`main.run(config)` orchestrates, in order:
1. `src/data_generator.py` → raw daily rows `RAW_COLUMNS` (date, location, precipitation_mm, temperature_mean_c, river_discharge_m3s). Cached in `data/raw/` and invalidated when schema/location set changes.
2. `src/preprocessing.clean_data` → `src/features.build_timeseries_features` (per-location, backward-looking features + `q_future`, `target_date`).
3. `preprocessing.time_split` → `compute_thresholds` (train only) → `apply_labels` → `features.add_threshold_features`.
4. `src/models.get_models` — ML models are `Pipeline(prep=ColumnTransformer, clf=...)`; baselines: `DummyBaseline`, `PersistenceBaseline`.
5. `src/benchmark` — CV with `preprocessing.time_series_cv_splits` (date-blocked expanding window + gap), test eval, per-location eval, `select_best_model` (CV metric, non-baseline).
6. `src/visualize` — all PNGs to `outputs/figures/`; metrics CSV/JSON to `outputs/metrics/`; model bundle (dict) to `outputs/models/best_model.joblib`.

## Leakage invariants — DO NOT BREAK (enforced by `tests/test_leakage.py`)
- Features at day t use only data ≤ t, computed within `groupby("location")`. `q_future`, `target_date`, `flood_risk` are never features (`features.NON_FEATURE_COLUMNS`).
- Split chronologically; train rows must have `target_date < test_start`.
- Thresholds come from train rows only; never use full-dataset stats (max, median, percentile) before the split.
- Imputation/scaling live inside the model Pipeline (fit per CV fold), never on the whole dataset.
- Select the best model by CV score, never by test score.
- No random/shuffled splits on this time-series data.

## Conventions
- Charts: fixed categorical `visualize.PALETTE` order (CVD-validated), no dual y-axes (use stacked panels), `matplotlib.use("Agg")`.
- Tests must stay offline: use the synthetic generator or monkeypatch `requests.get` / fetchers.
- Known data caveat: GloFAS snaps coordinates to the nearest ~5 km cell; Hue, Hanoi and ThuDuc currently hit small streams (mean discharge ≈ 2–3 m³/s). Check the printed grid coords when changing locations.
