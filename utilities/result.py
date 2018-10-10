from sklearn.metrics import precision_recall_fscore_support
from graphics import confusion_matrix as cm
from utilities import labels as lbs
import numpy as np


def calculate_support(y=None, test=None, print_confusion_matrix=False):
    """
        Calculate Precision, Recall, F-Measure
        :param y: Predicted classes
        :param test:
        :param print_confusion_matrix:
        :return: Precision, Recall, F-Measure
    """
    y_true = []
    y_pred = []
    labels = set()
    for i in range(len(test)):
        y_pred.append(test[i][:3])
        max = np.argmax(y[i])
        y_true.append(lbs.revert_labels[str(max)])
        labels.add(lbs.revert_labels[str(max)])
    scores = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    if print_confusion_matrix:
        cm.ConfusionMatrix(y_true, y_pred, list(labels))
    return scores[0], scores[1], scores[2]

