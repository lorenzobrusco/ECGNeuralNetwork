from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, MaxPooling2D, Activation
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
from random import randint
import keras
from keras import regularizers
import numpy as np
import random
import cv2
import json
import imutils

_train_dir = '../Data/dataset/train'
_validation_dir = '../Data/dataset/validation'

_train_dir_no_augmentation = '../Data/dataset_no_augmentation/train'
_validation_dir_no_augmentation = '../Data/dataset_no_augmentation/validation'

_labels_to_float = '{ "NOR": "0", "PVC" : "1", "PAB": "2", "LBB": "3", "RBB": "4", "APC": "5", "VFW": "6", "VEB": "7" }'
_labels = json.loads(_labels_to_float)

_train_files = 71207
_validation_files = 36413
_rotate_range = 30
_size = (64, 64)
_batch_size = 32
_epochs = 10
_n_classes = 8
_regularizers_l1 = 0.0001


def create_model_old():
    """
        Create model
        :param img_size:
        :return:
    """

    model = Sequential()

    model.add(
        Conv2D(64, (3, 3), strides=(1, 1), input_shape=(_size[0], _size[1], 1),
               kernel_regularizer=regularizers.l1(_regularizers_l1)))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_regularizer=regularizers.l1(0.0001)))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_regularizer=regularizers.l1(0.0001)))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_regularizer=regularizers.l1(0.0001)))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_regularizer=regularizers.l1(0.0001)))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_regularizer=regularizers.l1(0.0001)))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, kernel_regularizer=regularizers.l1(0.0001)))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_model():
    """
        Create model
        :param img_size:
        :return:
    """

    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(_size[0], _size[1], 1), activation='relu',
                     kernel_regularizer=regularizers.l1(_regularizers_l1)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1(_regularizers_l1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                     kernel_regularizer=regularizers.l1(_regularizers_l1)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(128, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1(_regularizers_l1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu',
                     kernel_regularizer=regularizers.l1(_regularizers_l1)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(256, (3, 3), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1(_regularizers_l1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l1(_regularizers_l1)))
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


def load_files(directory, shuffle=True, all=True):
    """
        Load each name file in the directory
        :param directory:
        :param shuffle:
        :return:
    """
    if all:
        files = [f for f in listdir(directory)]
    else:
        files = [f for f in listdir(directory) if f[-5] == '0']
    return files


def load_dataset(files, directory, batch_size, size, shuffle, random_crop, random_rotate, flip):
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
        if shuffle:
            random.shuffle(files)
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X, Y = image_to_array(files[batch_start:limit], size, directory, random_crop, random_rotate, flip)

            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size


def image_to_array(files, size, directory, random_crop, random_rotate, flip):
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
            random = list(file)
            random[-5] = str(randint(0, 9))
            random = "".join(random)
            file_name = directory + '/' + random
        else:
            file_name = directory + '/' + file

        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if random_rotate:
            img = imutils.rotate(img, randint(-_rotate_range, _rotate_range))
        if flip:
            img = cv2.flip(img, randint(-1, 1))
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


def training(model, shuffle=False, augmentation=True):
    """
        Training and testing the model
        :return:
    """

    train = load_files(_train_dir, shuffle=shuffle, all=False)
    validation = load_files(_validation_dir, shuffle=shuffle, all=False)

    chk = ModelCheckpoint("model.h5", monitor='val_loss', save_best_only=False)
    tsb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                      write_grads=False,
                                      write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                      embeddings_metadata=None, embeddings_data=None)
    eas = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                                        baseline=None)
    callbacks_list = [chk, tsb, eas]

    model.fit_generator(
        load_dataset(train, _train_dir, _batch_size, _size,
                     shuffle=augmentation,
                     random_crop=augmentation,
                     random_rotate=augmentation,
                     flip=augmentation),
        steps_per_epoch=steps(train, _batch_size),
        epochs=_epochs,
        validation_data=load_dataset(validation, _validation_dir, _batch_size, _size,
                                     shuffle=augmentation,
                                     random_crop=augmentation,
                                     random_rotate=augmentation,
                                     flip=augmentation),
        validation_steps=steps(validation, _batch_size),
        callbacks=callbacks_list)

    # model.save('model.h5')


def training_old(model, shuffle=False):
    """
        Training and testing the model
        :return:
    """

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_set = train_datagen.flow_from_directory(
        _train_dir_no_augmentation,
        target_size=_size,
        color_mode='grayscale',
        batch_size=_batch_size,
        shuffle=shuffle,
        class_mode='categorical')

    test_set = test_datagen.flow_from_directory(
        _validation_dir_no_augmentation,
        target_size=_size,
        color_mode='grayscale',
        batch_size=_batch_size,
        shuffle=shuffle,
        class_mode='categorical')

    model.fit_generator(
        train_set,
        steps_per_epoch=_train_files / _batch_size,
        epochs=25,
        validation_data=test_set,
        validation_steps=_validation_files / _batch_size)

    model.save('model_no_augmentation.h5')
