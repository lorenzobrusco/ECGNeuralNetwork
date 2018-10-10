from __future__ import print_function
from sklearn.metrics import confusion_matrix


class ConfusionMatrix():
    def __init__(self, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred)
        self.print_cm(cm, labels)

    def print_cm(self, cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
        """pretty print for confusion matrixes"""
        labels = list(labels)
        columnwidth = max([len(x) for x in labels] + [10])  # 5 is value length
        empty_cell = " " * columnwidth
        # Print header
        print("    " + empty_cell, end='')
        for label in labels:
            print("%{0}s".format(columnwidth) % label, end='')
        print()
        # Print rows
        for i, label1 in enumerate(labels):
            print("    %{0}s".format(columnwidth) % label1, end='')
            for j in range(len(labels)):
                cell = "%{0}.1f".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end='')
            print()
