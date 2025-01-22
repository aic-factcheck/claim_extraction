import matplotlib.pyplot as plt
from sklearn import metrics
from typing import Literal


def compute_conf_matrices(y_true: list, y_pred: list, labels: list, xticks_rotation: Literal['vertical', 'horizontal'] = 'vertical', cmap: str = "Blues", title: str = "conf_mats"):
    """
    Outputs one figure containing 3 confusion matrices for the same `y_true` and `y_pred` data with labels `labels`. 
    The first matrix shows counts, the second shows percentage normalized accros all values and the third shows percentage normalized accros rows.
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))
    fig.suptitle(title)

    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, labels=labels, xticks_rotation=xticks_rotation, ax=axs[0], cmap=cmap)
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, labels=labels, normalize='all', xticks_rotation=xticks_rotation, values_format=".2%", ax=axs[1], cmap=cmap)
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, labels=labels, normalize='true', xticks_rotation=xticks_rotation, values_format=".2%", ax=axs[2], cmap=cmap)

    return fig


def save_conf_matrices(y_true: list, y_pred: list, labels: list, fname: str, xlabel:str, ylabel:str, format: str, xticks_rotation: Literal['vertical', 'horizontal'] = "vertical", cmap: str = "Blues"):
    """
    Saves confusion matrices for usage in thesis.
    """
    fig = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, labels=labels, xticks_rotation=xticks_rotation, cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname=f"{fname}.{format}", bbox_inches='tight')

    fig = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, labels=labels, normalize='all', xticks_rotation=xticks_rotation, cmap=cmap, values_format=".2%")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname=f"{fname}_percentage.{format}",bbox_inches='tight')

    fig = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, labels=labels, normalize='true', xticks_rotation=xticks_rotation, cmap=cmap, values_format=".2%")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname=f"{fname}_percentage_rows.{format}", bbox_inches='tight')
