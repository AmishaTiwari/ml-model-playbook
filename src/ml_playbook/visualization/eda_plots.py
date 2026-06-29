# eda_plots.py

import matplotlib.pyplot as plt


def plot_bar(
    values,
    labels=None,
    annotations=None,
    title="",
    xlabel="",
    ylabel="",
    figsize=(6, 5),
    fmt=".1f",
    annotation_suffix=""
):
    """
    Create a bar plot with optional value annotations.

    Parameters
    ----------
    values : array-like
        Heights of the bars.

    labels : array-like, default=None
        Labels for the x-axis.

    annotations : array-like, default=None
        Values to display above bars. If None, bar heights are used.

    title : str, default=""
        Plot title.

    xlabel : str, default=""
        X-axis label.

    ylabel : str, default=""
        Y-axis label.

    figsize : tuple, default=(6, 5)
        Figure size.

    fmt : str, default=".1f"
        Format string for annotations.

    annotation_suffix : str, default=""
        Suffix appended to annotation text (e.g. "%").
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Use pandas plotting
    values.plot(
        kind="bar",
        ax=ax
    )

    if labels is not None:
        ax.set_xticklabels(labels)

    if annotations is None:
        annotations = values

    for i, annotation in enumerate(annotations):

        height = (
            values.iloc[i]
            if hasattr(values, "iloc")
            else values[i]
        )

        ax.text(
            i,
            height,
            f"{annotation:{fmt}}{annotation_suffix}",
            ha="center",
            va="bottom"
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.show()
    

def plot_categorical_distribution(
    data,
    feature,
    figsize=(5, 4)
):
    """
    Plot the distribution of a categorical feature with
    percentage annotations.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.

    feature : str
        Categorical feature to visualize.

    figsize : tuple, default=(5, 4)
        Figure size.
    """

    counts = (
        data[feature]
        .value_counts(dropna=False)
    )

    percentages = (
        data[feature]
        .value_counts(
            normalize=True,
            dropna=False
        )
        .mul(100)
    )

    plot_bar(
        values=counts,
        annotations=percentages,
        title=feature,
        ylabel="Count",
        figsize=figsize,
        fmt=".1f",
        annotation_suffix="%"
    )