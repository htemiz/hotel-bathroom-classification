from pandas import DataFrame
from os.path import join, exists
from os import makedirs, listdir
import pandas as pd
import numpy as np
import locale
from os.path import join, exists, abspath, dirname, basename, splitext

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.preprocessing import label_binarize

import seaborn as sns
from matplotlib import pyplot as plt




def plot_confusion_matrix(cm, title=None, on_screen=True, plot=True, save=False, labels=None, dpi=400):
    """
    https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics

    """
    if title is None:
        title = 'Confusion Matrix'

    if on_screen:
        print(cm)

    # labels = ["T" + str(x) for x in list(range(1, 6))]

    fig, ax = plt.subplots(dpi=dpi)
    sns.heatmap(cm, annot=True, fmt='g', cmap=plt.cm.Blues, ax=ax) # cmap='Spectral'
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.yaxis.set_ticklabels(labels)
    ax.xaxis.set_ticklabels(labels)

    if plot:
        plt.show()

    if save:
        plt.savefig(title + '_confusion_matrix.png',)

    return fig, ax


def plot_roc_curve(fpr, tpr, roc_auc, n_classes=5, save=False, plot=True, titles=None, labels=None, xlim=None, ylim=None, dpi=400):
    fig = plt.figure(dpi=dpi)
    legend_texts= list()

    linestyle ="-"
    lw = 2.25

    for i in range(len(fpr)):

        if i < n_classes:
            key=i

            if  labels is not None and n_classes == len(labels):
                nam = str(labels[i])
            else:
                nam = None

            label = nam + " (AUC = %0.3f)" % roc_auc[key]
        else:
            key = list(fpr)[i]
            label = "All_" + key +  " (AUC = %0.3f)" % roc_auc[key]

        if key in ('micro',):
            linestyle = ':'
        elif key in ('macro',):
            linestyle = '--'

        plt.plot(
            fpr[key],
            tpr[key],
            # color="darkorange",
            linestyle=linestyle,
            lw=lw,
            label=label,
        )

        plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--") #

        if xlim is not None:
            plt.xlim(0, 0.2)
        else:
            plt.xlim([-0.02, 1.02])

        if ylim is not None:
            plt.ylim(0.8, 1)
        else:
            plt.ylim([-0.02, 1.02])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        if titles is not None:
            plt.title(titles[key])
        else:
            plt.title("ROC Curve ")

    plt.legend(loc="lower right", title="Classes")

    if save:
        plt.savefig("ROC Curve.png")
    # plt.tight_layout()
    if plot:
        plt.show()

    return fig



def roc_macro(fpr, tpr, roc_auc, n_classes=5 ):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    :param fpr:
    :param tpr:
    :param roc_auc:
    :param n_classes:
    :return:
    """
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in reversed(range(n_classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])  # Dikkat: nan değerler için toplam sonucu nan oluyor.

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def roc_micro(y_test, y_score, n_classes=5):

    """

    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    :param y_test:
    :param y_score:
    :param n_classes:
    :return:
    """

    fpr = dict()
    tpr = dict()  # recall
    roc_auc = dict()
    pos_labels = dict()

    # ndim=1 ise, etiketler tek boyutlu verilmiş demektir.
    if y_test.ndim ==1 and n_classes >2:
        y_test= label_binarize(y_test, classes=np.arange(n_classes))

    # ndim=1 ise, etiketler tek boyutlu verilmiş demektir.
    if y_score.ndim ==1 and n_classes >2:
        y_score= label_binarize(y_score, classes=np.arange(n_classes))

    # micro-average ROC and area
    for i in range(n_classes):
        fpr[i], tpr[i], pos_labels[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], pos_labels['micro']=\
        roc_curve(y_test.ravel(), y_score.ravel())

    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    return fpr, tpr, pos_labels, roc_auc

