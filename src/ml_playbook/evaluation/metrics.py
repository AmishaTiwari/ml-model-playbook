# metrics.py

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_classification_model(
    y_true,
    y_pred,
    y_prob,
    decimals=3
):
    """
    Compute common classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted class labels.

    y_prob : array-like
        Predicted probabilities for the positive class.

    decimals : int, default=3
        Number of decimal places to round metric values.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metric names and scores.
    """

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "ROC-AUC",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score"
            ],
            "Score": [
                roc_auc_score(y_true, y_prob),
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred),
                recall_score(y_true, y_pred),
                f1_score(y_true, y_pred)
            ]
        }
    )

    metrics_df["Score"] = metrics_df["Score"].round(decimals)

    return metrics_df