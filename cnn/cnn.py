from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Dropout, MaxPooling2D
from graphics.train_val_tensorboard import TrainValTensorBoard
from keras import Sequential
from os import listdir
from utilities import labels as lbs
from random import randint
from keras import regularizers
from graphics import confusion_matrix as cm
import numpy as np
import random
import cv2
import imutils

_dataset_dir = '../Data/dataset_filtered/'
_model = 'models/cnn_model.h5'

_train_files = 71207
_validation_files = 36413
_rotate_range = 180
_size = (64, 64)
_batch_size = 32
_filters= (4, 4)
_epochs = 30
_n_classes = 8
_regularizers = 0.0001
_split_validation_percentage = 0.70
_split_test_percentage = 0.50
_probability_to_change = 0.30
_seed = 7


def create_model():
    """
        Create model
        :param img_size:
        :return:

    """

    model = Sequential()

    model.add(Conv2D(64, _filters, input_shape=(_size[0], _size[1], 1), padding='same',
                     kernel_regularizer=regularizers.l1_l2(_regularizers, _regularizers), activation='relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(64, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(
        Conv2D(128, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(
        Conv2D(128, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(
        Conv2D(256, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(
        Conv2D(256, _filters, kernel_regularizer=regularizers.l2(_regularizers), padding='same',
               activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(_regularizers), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def encode_label(file):
    """
        Encode the class label
        :param file:
        :return:
    """
    label = [0 for _ in range(_n_classes)]
    label[int(lbs.labels[file[:3]])] = 1
    return label


def image_to_array(files, size, directory, random_rotate, flip, encode_labels=True):
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
        img /= 255
        img = np.reshape(img, [size[0], size[1], 1])
        if encode_labels:
            label = encode_label(file)
        else:
            label = lbs.labels[file[:3]]
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
    with open('dataset/name_files_test.txt', 'w') as f:
        for item in test:
            f.write("%s\n" % item)
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
    train, validation, test = load_files(_dataset_dir)
    print(model.summary())
    callbacks_list = [ModelCheckpoint(_model, monitor='val_loss', save_best_only=True),
                      TrainValTensorBoard(write_graph=False)]

    model.fit_generator(
        load_dataset(train, _dataset_dir, _batch_size, _size, random_crop=augmentation, random_rotate=augmentation,
                     flip=augmentation),
        steps_per_epoch=steps(train, _batch_size), epochs=_epochs,
        validation_data=load_dataset(validation, _dataset_dir, _batch_size, _size, random_crop=augmentation,
                                     random_rotate=augmentation, flip=augmentation),
        validation_steps=steps(validation, _batch_size),
        callbacks=callbacks_list)

    evaluate_model(test, model)
    predict_model(test, model)


def evaluate_model(test, model):
    """
        Evaluate model
        :return:
    """
    x, y = image_to_array(test, _size, _dataset_dir, True, True, True)
    score = model.evaluate(x, y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def predict_model(test, model, confusion_matrix=False):
    """
        Predict model
        :return:
    """
    x, y = image_to_array(test, _size, _dataset_dir, True, True, True)
    y = model.predict_classes(np.reshape(x, (len(test), 64, 64, 1)))
    y_true = []
    y_pred = []
    labels = set()
    acc = 0
    for i in range(len(test)):
        y_pred += test[i][:3]
        y_true += lbs.revert_labels[str(y[i])]
        labels.add(lbs.revert_labels[str(y[i])])
        labels.add(test[i][:3])
        if y_true[i] == y_pred[i]:
            acc += 1
    acc /= len(test)
    print('Accuracy: %s' % acc)
    if confusion_matrix:
        labels = [lab for lab in labels]
        cm.ConfusionMatrix(y_true, y_pred, labels)
