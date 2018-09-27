from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Dropout, MaxPooling2D, Activation
from keras.optimizers import Nadam, SGD, Adam
from graphics.train_val_tensorboard import TrainValTensorBoard
from keras.models import load_model
from sklearn import preprocessing
from keras import Sequential
from os import listdir
from random import randint
from keras import regularizers
from graphics import confusion_matrix as cm
import numpy as np
import random
import cv2
import json
import imutils

_dataset_dir = '../Data/dataset/'
_model = 'cnn_model.h5'

_labels_to_float = '{ "NOR": "0", "PVC" : "1", "PAB": "2", "LBB": "3", "RBB": "4", "APC": "5", "VFW": "6", "VEB": "7" }'
_float_to_labels = '{ "0": "NOR", "1" : "PVC", "2": "PAB", "3": "LBB", "4": "RBB", "5": "APC", "6": "VFW", "7": "VEB" }'
_labels = json.loads(_labels_to_float)
_revert_labels = json.loads(_float_to_labels)

_train_files = 71207
_validation_files = 36413
_rotate_range = 180
_size = (64, 64)
_batch_size = 32
_epochs = 100
_n_classes = 8
_regularizers = 0.0001
_split_validation_percentage = 0.70
_split_test_percentage = 0.50
_probability_to_change = 0.30
_seed = 7

np.random.seed(_seed)
random.seed(_seed)


def create_model():
    """
        Create model
        :param img_size:
        :return:

    """

    model = Sequential()

    model.add(Conv2D(64, (5, 5), input_shape=(_size[0], _size[1], 1),
                     kernel_regularizer=regularizers.l1_l2(_regularizers, _regularizers), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), kernel_regularizer=regularizers.l2(_regularizers), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(Conv2D(128, (5, 5), kernel_regularizer=regularizers.l2(_regularizers), activation='relu'))
    model.add(Conv2D(128, (5, 5), kernel_regularizer=regularizers.l2(_regularizers), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(Conv2D(256, (5, 5), kernel_regularizer=regularizers.l2(_regularizers), activation='relu'))
    model.add(Conv2D(256, (5, 5), kernel_regularizer=regularizers.l2(_regularizers), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(_regularizers), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sgd = SGD(lr=0.1, momentum=0.8, decay=0.1 / _epochs, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def encode_label(file):
    """
        Encode the class label
        :param file:
        :return:
    """
    label = [0 for _ in range(_n_classes)]
    label[int(_labels[file[:3]])] = 1
    return label


def image_to_array(files, size, directory, random_crop, random_rotate, flip, encode_labels=True):
    """
        Convert an image to array and encode its label
        :param files:
        :param size:
        :param directory:
        :return: image converted and its label
    """
    images = []
    labels = []
    for file in files:
        if random_crop:
            if random.uniform(0, 1) < _probability_to_change:
                random_file = list(file)
                random_file[-5] = str(randint(0, 9))
                random_file = "".join(random_file)
                file_name = directory + '/' + random_file
            else:
                file_name = directory + '/' + file
        else:
            file_name = directory + '/' + file

        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if random_rotate:
            if random.uniform(0, 1) < _probability_to_change:
                img = imutils.rotate(img, randint(-_rotate_range, _rotate_range))
        if flip:
            if random.uniform(0, 1) < _probability_to_change:
                img = cv2.flip(img, randint(-1, 1))
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
        img = img.astype('float64')
        img = preprocessing.MinMaxScaler().fit_transform(img)
        img = np.reshape(img, [size[0], size[1], 1])
        if encode_labels:
            label = encode_label(file)
        else:
            label = _labels[file[:3]]
        images.append(img)
        labels.append(label)

    X = np.array(images)
    Y = np.array(labels)

    return X, Y


def steps(files, batch_size):
    """
        Calculate the number steps necessary to process each files
        :param files:
        :param batch_size:
        :return: the numbers of files divided to batch
    """
    return len(files) / batch_size


def load_files(directory):
    """
        Load each name file in the directory
        :param directory:
        :param shuffle:
        :return:
    """
    train = []
    validation = []
    test = []

    classes = {'NOR', 'PVC', 'PAB', 'LBB', 'RBB', 'APC', 'VFW', 'VEB'}

    classes_dict = dict()

    for key in classes:
        classes_dict[key] = [f for f in listdir(directory) if key in f if f[-5] == '0']
        random.shuffle(classes_dict[key])

    for _, item in classes_dict.items():
        train += item[: int(len(item) * _split_validation_percentage)]
        val = item[int(len(item) * _split_validation_percentage):]
        validation += val[: int(len(val) * _split_test_percentage)]
        test += val[int(len(val) * _split_test_percentage):]

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)
    return train, validation, test


def load_dataset(files, directory, batch_size, size, random_crop, random_rotate, flip):
    """
        Load dataset in minibatch
        :param directory:
        :param batch_size:
        :param size:
        :return:
    """

    L = len(files)

    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X, Y = image_to_array(files[batch_start:limit], size, directory, random_crop, random_rotate, flip)
            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size


def training(augmentation=True):
    """
        Training and testing the model
        :return:
    """
    model = create_model()
    train, validation, _ = load_files(_dataset_dir)

    callbacks_list = [ModelCheckpoint(_model, monitor='val_loss', save_best_only=True),
                      TrainValTensorBoard(write_graph=False)]

    model.fit_generator(
        load_dataset(train, _dataset_dir, _batch_size, _size,
                     random_crop=augmentation,
                     random_rotate=augmentation,
                     flip=augmentation),
        steps_per_epoch=steps(train, _batch_size),
        epochs=_epochs,
        validation_data=load_dataset(validation, _dataset_dir, _batch_size, _size,
                                     random_crop=augmentation,
                                     random_rotate=augmentation,
                                     flip=augmentation),
        validation_steps=steps(validation, _batch_size),
        callbacks=callbacks_list)


def evaluate_model():
    """
        Evaluate model
        :return:
    """
    model = load_model(_model)
    _, _, test = load_files(_dataset_dir)
    x, y = image_to_array(test, (64, 64), _dataset_dir, True, True, True)
    score = model.evaluate(x, y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def predict_model(confusion_matrix=False):
    """
        Predict model
        :return:
    """
    model = load_model(_model)
    _, _, test = load_files(_dataset_dir)
    x, y = image_to_array(test, (64, 64), _dataset_dir, True, True, True)
    y = model.predict_classes(np.reshape(x, (len(test), 64, 64, 1)))
    y_true = []
    y_pred = []
    labels = set()
    acc = 0
    for i in range(len(test)):
        y_pred += test[i][:3]
        y_true += _revert_labels[str(y[i])]
        labels.add(_revert_labels[str(y[i])])
        labels.add(test[i][:3])
        if y_true[i] == y_pred[i]:
            acc += 1
    acc /= len(test)
    print('Accuracy: %s' % acc)
    if confusion_matrix:
        labels = [lab for lab in labels]
        cm.ConfusionMatrix(y_true, y_pred, labels)
