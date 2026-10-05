"""Basic per-target metrics and equal-target aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _correlation(function, truth: np.ndarray, prediction: np.ndarray) -> float:
    if truth.size < 2 or np.all(truth == truth[0]) or np.all(prediction == prediction[0]):
        return float("nan")
    return float(function(truth, prediction).statistic)


def target_metrics(
    predictions: pd.DataFrame,
    *,
    group_col: str = "target_id",
    truth_col: str = "targets",
    prediction_col: str = "preds",
    uid_col: str = "query_uid",
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Use every row; undefined correlations remain NaN in the macro mean.

    RMSE retains the input activity units. Invalid data raises an error rather
    than being removed. A constant or singleton target has undefined correlation;
    its presence is counted and makes the corresponding macro correlation NaN.
    """
    required = {uid_col, group_col, truth_col, prediction_col}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    if predictions.empty:
        raise ValueError("Prediction table is empty")
    for column in (uid_col, group_col):
        values = predictions[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identifiers in {column!r}")
    if predictions[uid_col].astype(str).duplicated().any():
        raise ValueError("Prediction table has duplicate query identifiers")
    work = predictions.copy()
    work[group_col] = work[group_col].astype(str)
    for column in (truth_col, prediction_col):
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {column!r}")
    rows = []
    for target, frame in work.groupby(group_col, sort=True, dropna=False):
        truth = frame[truth_col].to_numpy(dtype=float)
        prediction = frame[prediction_col].to_numpy(dtype=float)
        rows.append(
            {
                group_col: str(target),
                "n": int(truth.size),
                "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                "pearson_r": _correlation(pearsonr, truth, prediction),
                "spearman_rho": _correlation(spearmanr, truth, prediction),
            }
        )
    per_target = pd.DataFrame(rows)
    summary = {
        "n_targets": int(len(per_target)),
        "n_samples": int(len(work)),
        "macro_rmse": float(np.mean(per_target["rmse"].to_numpy())),
        "macro_pearson_r": float(np.mean(per_target["pearson_r"].to_numpy())),
        "macro_spearman_rho": float(np.mean(per_target["spearman_rho"].to_numpy())),
        "undefined_pearson_targets": int(per_target["pearson_r"].isna().sum()),
        "undefined_spearman_targets": int(per_target["spearman_rho"].isna().sum()),
    }
    return per_target, summary
