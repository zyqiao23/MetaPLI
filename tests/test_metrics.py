import math

import numpy as np
import pandas as pd
import pytest

from metapli.metrics import target_metrics


def test_equal_target_mean_uses_all_rows():
    frame = pd.DataFrame({"query_uid": ["a", "b", "c", "d", "e"], "target_id": ["A", "A", "B", "B", "B"], "targets": [1, 2, 1, 2, 3], "preds": [2, 3, 3, 4, 5]})
    per_target, summary = target_metrics(frame)
    assert per_target["n"].tolist() == [2, 3]
    assert summary["n_samples"] == 5
    assert summary["macro_rmse"] == 1.5
    assert summary["macro_pearson_r"] == pytest.approx(1)
    assert summary["macro_spearman_rho"] == pytest.approx(1)


def test_undefined_correlations_do_not_silently_remove_targets():
    frame = pd.DataFrame({"query_uid": ["a", "b", "c"], "target_id": ["A", "A", "B"], "targets": [1, 2, 3], "preds": [1, 2, 4]})
    per_target, summary = target_metrics(frame)
    assert len(per_target) == 2
    assert summary["undefined_pearson_targets"] == 1
    assert summary["undefined_spearman_targets"] == 1
    assert math.isnan(summary["macro_pearson_r"])
    assert math.isnan(summary["macro_spearman_rho"])
    assert summary["macro_rmse"] == 0.5


@pytest.mark.parametrize("column,value", [("targets", np.nan), ("preds", np.inf), ("target_id", None), ("query_uid", "")])
def test_invalid_rows_raise_instead_of_being_removed(column, value):
    frame = pd.DataFrame({"query_uid": ["a", "b"], "target_id": ["A", "A"], "targets": [1.0, 2.0], "preds": [1.0, 2.0]})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        target_metrics(frame)


def test_duplicate_predictions_raise():
    frame = pd.DataFrame({"query_uid": ["a", "a"], "target_id": ["A", "A"], "targets": [1.0, 2.0], "preds": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        target_metrics(frame)
