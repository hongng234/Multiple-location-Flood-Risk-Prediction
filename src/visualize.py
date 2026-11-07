"""All pipeline charts (static PNGs). Colors follow a fixed, CVD-validated categorical order.

CVD-validated = the colors stay distinguishable for people with color-vision deficiency.
Every function takes the data to draw plus an output `path`, and writes one PNG file.
"""
from pathlib import Path   # file paths

import matplotlib          # plotting library

matplotlib.use("Agg")  # headless backend: works in terminals, CI and pytest

import matplotlib.pyplot as plt   # figure/axes API used by every chart below
import numpy as np                # arrays and averages
import pandas as pd               # DataFrames and timestamps
import seaborn as sns             # nicer default styling + heatmap helper
from sklearn.metrics import (     # ready-made curve/metric helpers for the evaluation charts
    ConfusionMatrixDisplay, auc, average_precision_score, precision_recall_curve, roc_curve,
)

sns.set_theme(style="whitegrid")   # white background with light grid lines for all charts

# Fixed categorical order (validated palette) — a series keeps its color across all charts
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8a85"      # baselines / reference lines
FLOOD = "#e34948"        # flood-day highlight
METRICS = ["precision", "recall", "f1", "roc_auc", "average_precision"]   # metrics shown in the comparison chart


def color_map(names) -> dict:
    """Map names to palette slots in their given (stable) order.

    How: the i-th name gets the i-th palette color; `%` wraps around if there are more names
    than colors. Because the order comes from config/model order, the same model or location
    gets the same color in every chart.
    """
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def _save(fig, path: str) -> None:
    """Save a figure to PNG and free its memory.

    How: create the folder if needed, save at 150 dpi with tight borders, then close the figure
    (open figures accumulate in memory when many charts are produced in one run).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# EDA charts
# ------------------------------------------------------------------
def plot_discharge_overview(raw: pd.DataFrame, thresholds: dict, test_start: str, path: str) -> None:
    """Line chart: full discharge history per location (small multiples) with threshold + test start.

    How: one subplot row per location (shared x-axis), each showing the discharge line, the
    train flood threshold (red dashed) and the test start date (gray dotted). Separate panels
    are used because rivers differ by orders of magnitude and would not fit on one y-axis.
    """
    locs = list(thresholds)      # location names, in config order
    colors = color_map(locs)     # {location: color}
    # One row of panels per location; squeeze=False keeps `axes` 2-D even with a single location
    fig, axes = plt.subplots(len(locs), 1, figsize=(12, 2.2 * len(locs)), sharex=True, squeeze=False)
    # Loop over locations: fill one panel per location with its own series and threshold
    for ax, loc in zip(axes[:, 0], locs):
        d = raw[raw["location"] == loc]   # rows of this location only
        ax.plot(d["date"], d["river_discharge_m3s"], color=colors[loc], lw=1.2)
        ax.axhline(thresholds[loc], color=FLOOD, ls="--", lw=1, label=f"Train P-threshold = {thresholds[loc]:.1f}")
        ax.axvline(pd.Timestamp(test_start), color=NEUTRAL, ls=":", lw=1.2, label="Test start")
        ax.set_title(loc, loc="left", fontsize=10)
        ax.set_ylabel("m³/s")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("River discharge per location (GloFAS)", x=0.01, ha="left")   # overall title, left-aligned
    fig.tight_layout()   # shrink spacing so labels don't overlap
    _save(fig, path)


def plot_class_distribution(df: pd.DataFrame, target: str, path: str) -> None:
    """Bar chart: number of flood vs no-flood days per location.

    How: count rows per (location, label), pivot labels into two columns, and draw grouped bars.
    """
    counts = df.groupby(["location", target]).size().unstack(fill_value=0)   # rows = location, columns = 0 / 1
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color=[PALETTE[0], FLOOD], edgecolor="white", linewidth=2, rot=15)
    ax.set_title("Flood risk class distribution per location (train + test)")
    ax.set_ylabel("Days")
    ax.legend(["No flood (0)", "Flood (1)"], frameon=False)
    _save(fig, path)


def plot_monthly_flood_frequency(df: pd.DataFrame, target: str, path: str) -> None:
    """Grouped bar chart: share of flood days per calendar month and location (seasonality).

    How: take the month of the target day, average the 0/1 label per (month, location)
    -> flood rate, convert to %, and draw one group of bars per month.
    """
    d = df.assign(month=df["target_date"].dt.month)   # add a month number column (1..12)
    # Mean of a 0/1 column = share of 1s; rows = month, columns = location, values in %
    rate = d.groupby(["month", "location"])[target].mean().unstack(fill_value=0) * 100
    locs = list(rate.columns)   # location names, for the colors
    fig, ax = plt.subplots(figsize=(12, 4.5))
    rate.plot(kind="bar", ax=ax, color=[color_map(locs)[l] for l in locs], edgecolor="white", linewidth=1, rot=0, width=0.8)
    ax.set_title("Flood-day frequency by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("% of days flagged flood")
    ax.legend(title="Location", frameon=False, ncol=min(len(locs), 5), fontsize=8)
    _save(fig, path)


def plot_correlation_heatmap(df: pd.DataFrame, path: str) -> None:
    """Heatmap of Pearson correlations between numeric features and the label (diverging, gray-free midpoint at 0).

    How: compute the correlation matrix of numeric columns and color it with a red-blue
    diverging map fixed to [-1, 1] so 0 is always white.
    """
    corr = df.corr(numeric_only=True)   # square table: correlation (-1..1) of every column pair
    fig, ax = plt.subplots(figsize=(13, 11))
    # annot=True writes each value in its cell; square=True keeps cells square
    sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 7}, cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, ax=ax, square=True)
    ax.set_title("Feature correlation heatmap (train set)")
    _save(fig, path)


# ------------------------------------------------------------------
# Benchmark charts
# ------------------------------------------------------------------
def plot_model_comparison(results: pd.DataFrame, path: str, title: str = "Model benchmark — test set") -> None:
    """Grouped bar chart: each model's test metrics.

    How: transpose the results table so metrics are the x-groups and models are the bars
    inside each group; the legend sits below the plot to avoid covering bars.
    """
    names = list(results["model"])   # model names, in table order (fixes their colors)
    fig, ax = plt.subplots(figsize=(11, 5))
    # .T (transpose): rows become metrics, columns become models -> one colored bar per model
    results.set_index("model")[METRICS].T.plot(
        kind="bar", ax=ax, rot=0, color=[color_map(names)[n] for n in names], edgecolor="white", linewidth=1.5, width=0.8)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)   # all metrics are between 0 and 1
    ax.legend(frameon=False, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8)
    _save(fig, path)


def plot_roc_curves(probas: dict, y_test, path: str) -> None:
    """ROC curves of all models on the test set.

    How: for each model compute (false-positive rate, true-positive rate) at every probability
    threshold and draw it; the diagonal is a random classifier.
    """
    colors = color_map(probas)   # iterating a dict gives its keys -> {model name: color}
    fig, ax = plt.subplots(figsize=(7, 6))
    # Loop over models: one ROC line each, with its AUC in the legend
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)   # `_` = the thresholds, not needed here
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls=":", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("ROC curves — test set")
    _save(fig, path)


def plot_pr_curves(probas: dict, y_test, path: str) -> None:
    """Precision–recall curves (more informative than ROC for rare floods) + prevalence line.

    How: for each model compute precision/recall at every threshold and draw a step curve;
    the horizontal line at the flood rate is what a random model would score.
    """
    colors = color_map(probas)   # {model name: color}
    fig, ax = plt.subplots(figsize=(7, 6))
    # Loop over models: one PR step curve each, with its average precision (PR-AUC) in the legend
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)   # precision and recall at every threshold
        ap = average_precision_score(y_test, p)             # area under this curve (one number)
        ax.step(rec, prec, where="post", color=colors[name], lw=2, label=f"{name} (AP = {ap:.3f})")
    prevalence = float(np.mean(y_test))   # share of flood days in the test set
    ax.axhline(prevalence, color=NEUTRAL, ls=":", lw=1, label=f"Random (prevalence {prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Precision–Recall curves — test set")
    _save(fig, path)


def plot_cv_folds(folds: pd.DataFrame, metric: str, path: str) -> None:
    """Line chart: CV score per fold (folds are chronological) — shows stability over time.

    How: x = fold number (later = more recent validation period), y = metric, one line per model.
    A flat line means the model is stable across years; a dip reveals a hard period.
    """
    names = list(folds["model"].unique())   # each model once, in first-seen order
    colors = color_map(names)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Loop over models: draw that model's score across folds
    for name in names:
        d = folds[folds["model"] == name]   # this model's rows (one per fold)
        ax.plot(d["fold"], d[metric], marker="o", markersize=6, lw=2, color=colors[name], label=name)
    ax.set_xticks(sorted(folds["fold"].unique()))   # whole-number ticks: 1, 2, 3, ...
    ax.set_xlabel("CV fold (later = more recent validation period)")
    ax.set_ylabel(metric)
    ax.set_title(f"Time-series CV — {metric} per fold")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_per_location_metrics(per_loc: pd.DataFrame, models: list[str], metric: str, path: str) -> None:
    """Grouped bar chart: one metric per location for selected models (e.g. best vs persistence).

    How: keep only the chosen models, pivot to (location × model), draw grouped bars and
    print each value on top of its bar. Colors come from the full model list, so a model
    keeps the same color as in the other benchmark charts.
    """
    # Rows = location, columns = the chosen models (in the given order), values = the metric
    d = per_loc[per_loc["model"].isin(models)].pivot(index="location", columns="model", values=metric)[models]
    all_models = list(per_loc["model"].unique())   # full model list -> same colors as other charts
    colors = color_map(all_models)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    d.plot(kind="bar", ax=ax, rot=15, color=[colors[m] for m in models], edgecolor="white", linewidth=2)
    # Loop over bar groups (one container per model): write the value above each bar
    for c in ax.containers:  # selective direct labels: values on bars (few bars only)
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Test {metric} per location")
    ax.set_ylabel(metric)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_confusion_matrix(y_test, y_pred, model_name: str, path: str) -> None:
    """Confusion matrix of the selected model on the test set.

    How: count (actual, predicted) pairs: hits, misses (actual flood, predicted none),
    false alarms and correct "no flood" days, then render them as a 2×2 grid.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    # Count and draw the 2×2 table (rows = actual, columns = predicted)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues", display_labels=["No flood", "Flood"])
    ax.set_title(f"Confusion matrix — {model_name}")
    ax.grid(False)   # the seaborn grid would draw lines through the cells
    _save(fig, path)


def plot_feature_importance(model, path: str, top_n: int = 20) -> None:
    """Horizontal bar chart: tree importances or |logistic coefficients| (names from the preprocessor).

    How: read the classifier inside the Pipeline, get its importance scores, label them with
    the preprocessor's output column names and plot the `top_n` largest.
    """
    # Baselines are not Pipelines and have nothing to explain: skip silently
    if not hasattr(model, "named_steps"):
        return
    clf = model.named_steps["clf"]                              # the trained classifier inside the Pipeline
    names = model.named_steps["prep"].get_feature_names_out()   # column names after preprocessing (incl. one-hot)
    # Tree models expose feature_importances_; linear models expose coef_ (use its magnitude);
    # any other classifier has no importance score, so no chart is drawn
    if hasattr(clf, "feature_importances_"):
        values, label = clf.feature_importances_, "Importance"
    elif hasattr(clf, "coef_"):
        values, label = np.abs(clf.coef_[0]), "|Coefficient| (standardized)"
    else:
        return
    names = [n.split("__", 1)[-1] for n in names]  # drop "num__" / "cat__" prefixes
    imp = pd.Series(values, index=names).sort_values().tail(top_n)   # the top_n largest, smallest first
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(imp) + 1.5))         # height grows with the number of bars
    imp.plot(kind="barh", ax=ax, color=PALETTE[0])
    ax.set_title(f"Feature importance — top {len(imp)}")
    ax.set_xlabel(label)
    _save(fig, path)


def plot_location_timeseries(test_df: pd.DataFrame, proba, threshold: float, location: str,
                             model_name: str, path: str) -> None:
    """Two stacked line panels (shared x, no dual axis) for one location on the test period:
    top = observed discharge on the target day vs threshold, bottom = predicted flood probability.
    Actual flood days are shaded red in both panels.

    How: attach the probabilities to the test rows (same order), keep one location, sort by
    target day, shade every real flood day, then draw discharge (top) and probability (bottom).
    """
    d = test_df.assign(proba=np.asarray(proba))                   # add the predicted probability column
    d = d[d["location"] == location].sort_values("target_date")   # one location, in time order
    # Two panels stacked vertically, sharing the date axis; the top one is taller (3:2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[3, 2])

    # Shade actual flood days in both panels: a 1-day-wide band centred on each flood day
    flood_days = d.loc[d["flood_risk"] == 1, "target_date"]   # dates where the label is 1
    for ax in (ax1, ax2):
        for day in flood_days:
            ax.axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12), color=FLOOD, alpha=0.12, lw=0)

    # Top panel: what really happened (discharge vs flood threshold)
    ax1.plot(d["target_date"], d["q_future"], color=PALETTE[0], lw=1.6, label="Observed discharge")
    ax1.axhline(threshold, color=FLOOD, ls="--", lw=1, label=f"Flood threshold ({threshold:.1f} m³/s)")
    ax1.set_ylabel("m³/s")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_title(f"{location} — test period: discharge vs forecast ({model_name})", loc="left")

    # Bottom panel: what the model predicted the day before (probability vs 0.5 alert line)
    ax2.plot(d["target_date"], d["proba"], color=PALETTE[1], lw=1.6, label="Predicted flood probability")
    ax2.axhline(0.5, color=NEUTRAL, ls=":", lw=1, label="Decision threshold 0.5")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("P(flood)")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, path)
