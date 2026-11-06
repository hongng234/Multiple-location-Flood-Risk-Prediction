Forecast **tomorrow's** flood risk for 5 locations in Vietnam from rainfall and river-discharge data, with a leakage-safe ML pipeline.

## 1. Quickstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python

.venv/Scripts/python main.py --refresh   # download data + train + benchmark
.venv/Scripts/python predict.py          # live forecast for tomorrow
.venv/Scripts/python -m pytest           # offline tests
```

With `make`: `make install`, `make refresh`, `make run`, `make predict`, `make test`, `make cov`, `make clean`.
*(Windows: install via `choco install make` / `scoop install make`, or use Git Bash.)*

## 2. What it does

1. **Download data**
   - Sources: daily rainfall and temperature from Open-Meteo Archive; river discharge from GloFAS via the Open-Meteo Flood API.
   - Scope: every location in `config.yaml`, 2020–2024.
2. **Build features**
   - Every feature looks only at the past: 3/7/14-day rain sums, the Antecedent Precipitation Index (API), discharge lags and trend, and seasonality (day-of-year sin/cos).
   - Features are computed separately for each location.
3. **Label**
   - `flood_risk = 1` if discharge on day *t+1* exceeds that location's P90.
   - P90 is computed on the **train period only**.
4. **Split and evaluate**
   - Chronological split: train 2020–2023, test 2024.
   - Time-series cross-validation (date-blocked folds with a gap).
   - Compares 2 baselines with 3 ML models.
5. **Save outputs**
   - Metrics and 16 charts.
   - A self-contained model bundle for `predict.py`.

## 3. Results (test 2024, horizon = 1 day)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| DummyBaseline | 0.000 | 0.000 | 0.000 | 0.500 | 0.133 |
| PersistenceBaseline | 0.781 | 0.781 | 0.781 | 0.874 | 0.639 |
| LogisticRegression | 0.585 | 0.950 | 0.724 | 0.975 | 0.863 |
| RandomForest | 0.708 | 0.851 | 0.773 | 0.975 | 0.882 |
| **GradientBoosting** (selected by CV PR-AUC) | 0.847 | 0.756 | **0.799** | 0.974 | **0.885** |

- **Why the numbers changed:** the older version reported ROC-AUC ≈ 0.999 because the label was computed from same-day discharge that was also a feature (target leakage). These numbers are honest.
- **The key comparison:** against **PersistenceBaseline** ("tomorrow = today"), which is already strong at a 1-day horizon.
- **Why PR-AUC:** only about 13% of days are flood days, so a model that always predicts "no flood" still gets 87% accuracy. PR-AUC measures how well flood days are actually detected.

## 4. Project layout

```
config.yaml            # all settings
main.py                # train + benchmark pipeline
predict.py             # live next-day forecast
src/
  config.py            # load YAML, create folders
  data_loader.py       # CSV I/O + Open-Meteo API clients
  data_generator.py    # raw dataset (open_meteo | synthetic)
  features.py          # leakage-safe time-series features
  preprocessing.py     # clean, time split, train-only labels, ColumnTransformer, CV folds
  models.py            # baselines + ML pipelines
  benchmark.py         # CV, test, per-location metrics, model selection
  visualize.py         # all charts
tests/                 # pytest suite (offline)
docs/PROJECT_GUIDE.md  # detailed guide
outputs/               # figures/, metrics/, models/ (generated)
data/                  # raw/, processed/ (generated)
```

## 5. Outputs

- `outputs/metrics/`: `cv_results.csv`, `cv_folds.csv`, `test_results.csv`, `per_location_results.csv`, `thresholds.json`.
- `outputs/figures/`:
  - Data exploration: discharge overview, class distribution, monthly flood frequency, correlation heatmap.
  - Model evaluation: model comparison, ROC, PR, CV folds, per-location F1, confusion matrix, feature importance.
  - Per location: `timeseries_<location>.png`.
- `outputs/models/best_model.joblib`: dict `{model, model_name, feature_columns, thresholds, horizon_days, features_config, trained_until}`.

## 6. How I built this (and where Claude Code helped)

The idea came first: use free, global Earth-observation and climate-model data (ERA5 rainfall, GloFAS river discharge) to warn about floods one day ahead for places in Vietnam. I planned the pipeline and wrote the first working version myself (data download, cleaning, features and config — commit `81b1edd`), with some help from AI coding assistants. I then brought in Claude Code to debug and audit that code and to build the rest: the benchmark with baselines, the leakage tests, the charts and the live `predict.py`. I made the design decisions and the plan, and I reviewed every change. Claude Code made auditing, testing and documentation faster.

1. **Plan mode audit**
   - Starting from my first version, I asked Claude Code to audit the pipeline for logic errors and data leakage before changing anything.
   - Finding: the label was computed from same-day discharge that was also a feature, so ROC-AUC ≈ 0.999 was not real.
2. **Reframing the task**
   - Switched to a next-day forecast: the label is discharge at *t+1* above the train-period P90.
   - Added a persistence baseline to keep the evaluation honest.
3. **Leakage rules as tests**
   - The rules are written in `CLAUDE.md`.
   - `tests/test_leakage.py` checks them automatically: features use only past data, the split is chronological, thresholds come from train data only, preprocessing stays inside the Pipeline, and the model is selected by CV score.
4. **Iterate and verify**
   - Every refactor was followed by the offline test suite (26 tests, about 15 s).
   - The full benchmark was re-run to confirm the results.
5. **Documentation**
   - Code comments, `CLAUDE.md` and the bilingual guide in `docs/` were drafted with Claude Code and then edited by me.

**Result:** after fixing the leakage, test PR-AUC is 0.885 for Gradient Boosting, against 0.639 for the persistence baseline.

## 7. Limitations

- **Labels are a proxy:** they are based on high GloFAS discharge, not on observed flood events.
- **GloFAS coordinates:** GloFAS snaps coordinates to the nearest ~5 km cell. **Hue, Hanoi and ThuDuc currently land on small streams** (mean ≈ 2–3 m³/s, versus thousands of m³/s for the Red River at Hanoi). Move the coordinates onto the main river channel for meaningful results.
- **Urban flooding:** flooding from heavy rain and poor drainage is not captured by river discharge.