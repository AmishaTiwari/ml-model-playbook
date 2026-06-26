import pandas as pd


def compare_classification_metrics(
    metrics: dict[str, pd.DataFrame],
    reference: str,
    decimals: int = 3
) -> pd.DataFrame:
    """
    Compare classification metrics across multiple workflows.

    Parameters
    ----------
    metrics : dict[str, pd.DataFrame]
        Dictionary mapping workflow names to metric DataFrames.
        Each DataFrame must contain the columns:
        - Metric
        - Score

    reference : str
        Name of the reference workflow used to compute differences.

    decimals : int, default=3
        Number of decimal places to round the output.

    Returns
    -------
    pd.DataFrame
        Comparison table containing all workflows and the difference
        between each workflow and the reference workflow.
    """

    comparison = pd.concat(
        [
            df.set_index("Metric")["Score"].rename(name)
            for name, df in metrics.items()
        ],
        axis=1
    ).reset_index()

    for name in metrics:
        if name != reference:
            comparison[f"Δ {reference} -> {name}"] = (
                comparison[name]
                - comparison[reference]
            )

    return comparison.round(decimals)