from utilities import result
from ann import ann
from cnn import cnn

'''
    Use this script such as main

    In order to see the the information about 
    the neural network performe this command:
    tensorboard --logdir=logs
'''


def combine_result(y1, y2, test, weighted=True, print_precision=True, print_confusion_matrix=True):
    """
        Combine the result of two prediction models
        :param y1:
        :param y2:
        :param test: Used to calculate the true value of test
        :return:
    """
    precision_y1, _, f_measure_y1 = result.calculate_support(y1, test)
    precision_y2, _, f_measure_y2 = result.calculate_support(y2, test)
    y_new = [[0 for _ in range(len(y_ann[0]))] for _ in range(len(y1))]

    for i in range(len(y1)):
        for j in range(len(y1[i])):
            if weighted:
                y_new[i][j] = (y1[i][j] * f_measure_y1 + y2[i][j] * f_measure_y2) / 2
            else:
                y_new[i][j] = (y1[i][j] + y2[i][j]) / 2

    if print_precision:
        print('[CONFUSION MATRIX]\n')
        precision_y_new, _, f_measure_y_new = result.calculate_support(y_new, test, print_confusion_matrix=print_confusion_matrix)
        print('[ACCURACY CNN]: %s' % precision_y1)
        print('[ACCURACY ANN]: %s' % precision_y2)
        print('[ACCURACY NEW]: %s' % precision_y_new)
    return y_new


def save_test(test):
    """
        Save the list of test file in a txt
        :param test:
        :return:
    """
    with open('dataset/test.txt', 'w') as f:
        for item in test:
            f.write('%s\n' % item)


def convert_dataset_to_ann(test, train=None, validation=None):
    """
        Convert file .png to .txt
        :param train:
        :param validation:
        :param test:
        :return:
    """
    if train is not None:
        train_ann = [train[i][:-5] + '.txt' for i in range(len(train))]
    else:
        train_ann = None
    if validation is not None:
        val_ann = [validation[i][:-5] + '.txt' for i in range(len(validation))]
    else:
        val_ann = None
    test_ann = [test[i][:-5] + '.txt' for i in range(len(test))]
    return train_ann, val_ann, test_ann


def read_test():
    """
        Return the list of test files
        :return:
    """

    with open('dataset/test.txt', 'r') as f:
        test = f.read().split('\n')
    return test


""" Load test set from file """
test_cnn = read_test()
_, _, test_ann = convert_dataset_to_ann(test_cnn)

""" Load ann and cnn models """
ann_model = ann.load_ann_model()
cnn_model = cnn.load_cnn_model()

""" Predict new classes and combine the answers """
y_ann = ann.predict_model(model=ann_model, test=test_ann)
y_cnn = cnn.predict_model(model=cnn_model, test=test_cnn)
y_new = combine_result(y_cnn, y_ann, test_ann, weighted=True, print_precision=True, print_confusion_matrix=True)
