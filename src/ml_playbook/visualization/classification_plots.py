# classification_plots.py

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title="Confusion Matrix",
    figsize=(5, 5)
):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted class labels.

    labels : list, optional
        Labels to display on the axes.

    title : str, default="Confusion Matrix"
        Plot title.

    figsize : tuple, default=(5, 5)
        Figure size.
    """

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    ).plot(ax=ax)

    ax.set_title(title)

    plt.show()