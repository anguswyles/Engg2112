"""
Train tabular models to predict future drought state windows.

This complements the drought-onset classifier. The onset model answers:
"will drought begin around t+h?". This script answers: "will the soil be in
drought during a later 24-hour window?". Running state windows sequentially
gives the building blocks for a practical drought-duration estimate.

Usage:
    python scripts/run_drought_state_prediction.py
    python scripts/run_drought_state_prediction.py "test 15 - drought state 168h"
    python scripts/run_drought_state_prediction.py --start-horizon 168 --max-horizon 336
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config import THRESHOLD, XGB_PARAMS
from prepare import build_xgboost_dataset, get_feature_cols, load_all_stations, station_temporal_split


IMAGES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
COLORS = {
    "Persistence": "black",
    "Linear Trend": "gray",
    "Random Forest": "steelblue",
    "XGBoost": "darkorange",
}


def next_test_folder_name() -> str:
    dirs = [
        d
        for d in os.listdir(IMAGES_ROOT)
        if os.path.isdir(os.path.join(IMAGES_ROOT, d)) and d.startswith("test ")
    ]
    nums: list[int] = []
    for d in dirs:
        try:
            nums.append(int(d.split(" ", 2)[1]))
        except (IndexError, ValueError):
            pass
    return f"test {max(nums) + 1 if nums else 1}"


def cap_rows(rng: np.random.Generator, *arrays, max_n: int | None = None):
    arrays = tuple(arrays)
    if max_n is None or len(arrays[0]) <= max_n:
        return arrays
    idx = rng.choice(len(arrays[0]), size=max_n, replace=False)
    idx.sort()
    return tuple(a[idx] for a in arrays)


def scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    neg = max(int((y == 0).sum()), 1)
    pos = max(int((y == 1).sum()), 1)
    return neg / pos


def threshold_sweeps(y_true: np.ndarray, prob: np.ndarray, max_false_alarm_rate: float):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = (0.5, -1.0)
    best_recall_at_fa = (0.5, -1.0, 1.0)
    rows = []
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        far = fp / (fp + tn) if (fp + tn) else 0.0
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        rows.append(
            {
                "Probability threshold": round(float(threshold), 4),
                "Recall": round(float(rec), 4),
                "Precision": round(float(prec), 4),
                "F1": round(float(f1), 4),
                "False alarm rate": round(float(far), 4),
            }
        )
        if f1 > best_f1[1]:
            best_f1 = (threshold, f1)
        if far <= max_false_alarm_rate and rec > best_recall_at_fa[1]:
            best_recall_at_fa = (threshold, rec, far)
    return pd.DataFrame(rows), float(best_f1[0]), float(best_recall_at_fa[0])


def evaluate_probs(model_name: str, y_true: np.ndarray, prob: np.ndarray, threshold: float, label: str) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "Model": model_name,
        "Threshold label": label,
        "Probability threshold": round(float(threshold), 4),
        "Test samples": int(len(y_true)),
        "Drought windows": int(y_true.sum()),
        "Caught": int(tp),
        "Missed": int(fn),
        "False alarms": int(fp),
        "Recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
        "False alarm rate": round(false_alarm_rate, 4),
        "PR AUC": round(average_precision_score(y_true, prob), 4),
        "ROC AUC": round(roc_auc_score(y_true, prob), 4) if len(np.unique(y_true)) == 2 else np.nan,
    }


def drought_state_labels(
    stations: dict,
    station_keys: np.ndarray,
    base_times: np.ndarray,
    lead_hours: int,
    window_hours: int,
    threshold: float,
    min_dry_hours: int,
) -> np.ndarray:
    """Label a future window as drought if enough hourly readings are below threshold."""
    series_cache = {
        key: pair[0]["sm_value"].ffill().bfill().dropna()
        for key, pair in stations.items()
        if "sm_value" in pair[0].columns
    }
    labels = np.zeros(len(station_keys), dtype=int)
    lead_delta = pd.Timedelta(hours=int(lead_hours))
    end_delta = pd.Timedelta(hours=int(window_hours))
    for idx, (station, base_time) in enumerate(zip(station_keys, base_times)):
        sm = series_cache.get(station)
        if sm is None:
            continue
        start = pd.Timestamp(base_time) + lead_delta
        end = start + end_delta
        window = sm.loc[start : end - pd.Timedelta(nanoseconds=1)]
        labels[idx] = int((window < threshold).sum() >= min_dry_hours) if len(window) else 0
    return labels


def trend_probability(
    current_sm: np.ndarray,
    lag_sm: np.ndarray,
    lead_hours: int,
    window_hours: int,
    lag_hours: int,
    threshold: float,
) -> np.ndarray:
    slope_per_hour = (current_sm - lag_sm) / float(lag_hours)
    window_midpoint = lead_hours + (window_hours / 2)
    projected = current_sm + slope_per_hour * float(window_midpoint)
    distance = threshold - projected
    return 1.0 / (1.0 + np.exp(-distance / 0.01))


def fit_models(X_train: np.ndarray, y_train: np.ndarray, args) -> dict[str, object]:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    xgb_params = {
        key: value for key, value in XGB_PARAMS.items() if key not in {"nthread", "random_state", "verbosity"}
    }
    xgb = XGBClassifier(
        **xgb_params,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight(y_train),
        random_state=args.seed,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    return {"Random Forest": rf, "XGBoost": xgb}


def run_window(args, stations: dict, lead_hours: int) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    # Need enough future soil-moisture observations to cover the full state window.
    build_horizon = lead_hours + args.window_hours - 1
    data = build_xgboost_dataset(stations, horizon=build_horizon, with_weather=args.weather)
    feature_cols = get_feature_cols(with_weather=args.weather, horizon=build_horizon)

    if args.currently_wet_only:
        data = data[data["sm_value"] >= args.threshold].copy()

    X = data[feature_cols].values
    y = drought_state_labels(
        stations=stations,
        station_keys=data["station"].values,
        base_times=data.index.values,
        lead_hours=lead_hours,
        window_hours=args.window_hours,
        threshold=args.threshold,
        min_dry_hours=args.min_dry_hours,
    )
    tr, te = station_temporal_split(data)
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    rng = np.random.default_rng(args.seed + lead_hours)
    X_train, y_train = cap_rows(rng, X_train, y_train, max_n=args.max_train_rows)

    current_sm = data["sm_value"].values[te]
    lag_col = f"sm_lag_{args.trend_lag_hours}h"
    if lag_col not in data.columns:
        raise SystemExit(f"Trend baseline needs {lag_col}; available lags are configured in models/config.py")
    lag_sm = data[lag_col].values[te]

    print(
        f"State window t+{lead_hours}h to t+{lead_hours + args.window_hours}h: "
        f"train={len(y_train):,} positives={int(y_train.sum()):,} "
        f"test={len(y_test):,} positives={int(y_test.sum()):,}"
    )

    prob_by_model = {
        "Persistence": (current_sm < args.threshold).astype(float),
        "Linear Trend": trend_probability(
            current_sm,
            lag_sm,
            lead_hours,
            args.window_hours,
            args.trend_lag_hours,
            args.threshold,
        ),
    }
    models = fit_models(X_train, y_train, args)
    prob_by_model.update({name: model.predict_proba(X_test)[:, 1] for name, model in models.items()})

    metric_rows = []
    sweep_frames = []
    for name, prob in prob_by_model.items():
        sweep, best_f1_threshold, best_recall_threshold = threshold_sweeps(
            y_test, prob, max_false_alarm_rate=args.max_false_alarm_rate
        )
        sweep.insert(0, "Lead hours", lead_hours)
        sweep.insert(1, "Window hours", args.window_hours)
        sweep.insert(2, "Model", name)
        sweep_frames.append(sweep)
        thresholds = [
            ("default_0.5", 0.5),
            ("best_f1", best_f1_threshold),
            (f"max_recall_at_fa<={args.max_false_alarm_rate}", best_recall_threshold),
        ]
        for label, threshold in thresholds:
            row = evaluate_probs(name, y_test, prob, threshold, label)
            metric_rows.append({"Lead hours": lead_hours, "Window hours": args.window_hours, **row})

    fi_rows = []
    if "XGBoost" in models:
        importances = pd.Series(models["XGBoost"].feature_importances_, index=feature_cols).sort_values(ascending=False)
        for feature, importance in importances.items():
            fi_rows.append(
                {
                    "Lead hours": lead_hours,
                    "Window hours": args.window_hours,
                    "Feature": feature,
                    "Importance": float(importance),
                }
            )

    del X, X_train, X_test, models
    gc.collect()
    return metric_rows, pd.concat(sweep_frames, ignore_index=True), pd.DataFrame(fi_rows)


def plot_summary(metrics: pd.DataFrame, out_dir: str) -> None:
    best = metrics[metrics["Threshold label"] == "best_f1"].copy()
    if best.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for model, group in best.groupby("Model"):
        color = COLORS.get(model, None)
        axes[0].plot(group["Lead hours"], group["Recall"], marker="o", label=model, color=color)
        axes[1].plot(group["Lead hours"], group["Precision"], marker="o", label=model, color=color)
        axes[2].plot(group["Lead hours"], group["F1"], marker="o", label=model, color=color)
    axes[0].set_ylabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[2].set_ylabel("F1")
    for ax in axes:
        ax.set_xlabel("Future state window start (hours)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[2].legend(fontsize=8, loc="lower left")
    plt.suptitle("Drought state prediction by future window (best-F1 threshold)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drought_state_best_f1_by_window.png"), dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default=None, help="Output folder under images/ (default: next test N)")
    parser.add_argument("--start-horizon", type=int, default=168, help="First future state window start in hours")
    parser.add_argument("--max-horizon", type=int, default=336, help="Last future state window start in hours")
    parser.add_argument("--step-hours", type=int, default=24, help="Gap between future state windows")
    parser.add_argument("--window-hours", type=int, default=24, help="Future state window length")
    parser.add_argument("--min-dry-hours", type=int, default=6, help="Dry hours required inside window for drought state")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Soil moisture drought threshold")
    parser.add_argument("--no-weather", action="store_true", help="Train without current/future weather inputs")
    parser.add_argument(
        "--all-current-states",
        action="store_true",
        help="Train/evaluate on all current soil states instead of only currently-wet rows",
    )
    parser.add_argument("--trend-lag-hours", type=int, default=24, help="Lag used for linear trend baseline")
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.05)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional cap for train rows per window")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.weather = not args.no_weather
    args.currently_wet_only = not args.all_current_states
    return args


def main() -> None:
    args = parse_args()
    run_name = args.name or next_test_folder_name()
    out_dir = os.path.join(IMAGES_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\nDrought state prediction")
    print(f"State windows: t+{args.start_horizon}h to t+{args.max_horizon}h every {args.step_hours}h")
    print(f"Window length: {args.window_hours}h")
    print(f"Dry-state rule: at least {args.min_dry_hours}h below {args.threshold}")
    print(f"Current rows: {'currently wet only' if args.currently_wet_only else 'all current states'}")
    print(f"Weather inputs: {'yes' if args.weather else 'no'}")
    print(f"Output: {out_dir}\n")

    stations = load_all_stations(with_weather=args.weather)
    all_metrics = []
    all_sweeps = []
    all_importance = []
    for lead_hours in range(args.start_horizon, args.max_horizon + 1, args.step_hours):
        metrics, sweep, importance = run_window(args, stations, lead_hours)
        all_metrics.extend(metrics)
        all_sweeps.append(sweep)
        all_importance.append(importance)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "drought_state_metrics.csv"), index=False)
    pd.concat(all_sweeps, ignore_index=True).to_csv(
        os.path.join(out_dir, "drought_state_threshold_sweep.csv"), index=False
    )
    pd.concat(all_importance, ignore_index=True).to_csv(
        os.path.join(out_dir, "xgboost_state_feature_importance.csv"), index=False
    )
    plot_summary(metrics_df, out_dir)

    print("\nDrought state metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved drought state outputs to {out_dir}")


if __name__ == "__main__":
    main()
