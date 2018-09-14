from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, MaxPooling2D, Activation
from keras import Sequential
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from os import listdir
import keras
import numpy as np
import random
import cv2
import json

_train_dir = '../Data/dataset/train'
_validation_dir = '../Data/dataset/validation'
_labels_to_float = '{ "NOR": "0", "PVC" : "1", "PAB": "2", "LBB": "3", "RBB": "4", "APC": "5", "VFW": "6", "VEB": "7" }'
_labels = json.loads(_labels_to_float)

_size = (64, 64)
_batch_size = 32
_epochs = 10
_n_classes = 8


def create_model():
    """
        Create model
        :param img_size:
        :return:
    """

    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(_size[0], _size[1], 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=8, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def normalize(vector):
    """
        Transform each values between [0, 1]
        :param vector:
        :return:
    """
    vector_normalized = [i / 255 for i in vector]
    return np.array(vector_normalized, dtype=float)


def encode_label(file):
    """
        Encode the class label
        :param file:
        :return:
    """
    label = [0 for _ in range(_n_classes)]
    label[int(_labels[file[:3]])] = 1
    return label


def load_files(directory, shuffle=True, augmentation=True):
    """
        Load each name file in the directory
        :param directory:
        :param shuffle:
        :return:
    """
    if augmentation:
        files = [f for f in listdir(directory)]
    else:
        files = [f for f in listdir(directory) if f[-5] == '0']
    if shuffle:
        random.shuffle(files)
    return files


def load_dataset(files, directory, batch_size, size):
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
            X, Y = image_to_array(files[batch_start:limit], size, directory)

            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size


def image_to_array(files, size, directory):
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
        img = cv2.imread(directory + '/' + file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
        img = np.reshape(img, [size[0], size[1], 1])
        img = normalize(img)
        label = encode_label(file)
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


def training(model):
    """
        Training and testing the model
        :return:
    """
    train = load_files(_train_dir, shuffle=True, augmentation=False)
    validation = load_files(_validation_dir, shuffle=True, augmentation=False)

    model.fit_generator(
        load_dataset(train, _train_dir, _batch_size, _size),
        steps_per_epoch=steps(train, _batch_size),
        epochs=_epochs,
        validation_data=load_dataset(validation, _validation_dir, _batch_size, _size),
        validation_steps=steps(validation, _batch_size))

    model.save('model.h5')
