from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Dense
from graphics.train_val_tensorboard import TrainValTensorBoard
from keras import Sequential
from dataset import dataset
from graphics import confusion_matrix as cm
from utilities import labels as lbs
import numpy as np

_dataset_dir = '../Data/dataset_ann/'
_model = '../Models/ann.h5'
_batch_size = 32
_epochs = 30
_n_classes = 8


def create_model():
    """
        Create model
        :return:
    """
    model = Sequential()
    model.add(Dense(100, input_dim=100, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def load_ann_model():
    """
        Load and return model
        :return:
    """
    model = load_model(_model)
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


def load_sequence(files, directory, encode_labels=True):
    """
        Convert an image to array and encode its label
        :param files:
        :return: image converted and its label
    """
    sequences = []
    labels = []
    for file in files:
        file_name = directory + '/' + file
        if encode_labels:
            label = encode_label(file)
        else:
            label = lbs.labels[file[:3]]
        with open(file_name, 'r') as f:
            sequence = f.read().split(',')
        sequences.append(sequence)
        labels.append(label)

    X = np.array(sequences)
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


def load_dataset(files, directory, batch_size):
    """
        Load dataset in minibatch
        :param directory:
        :param batch_size:
        :return:
    """

    L = len(files)

    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X, Y = load_sequence(files[batch_start:limit], directory)
            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size


def training(train=None, validation=None):
    """
        Training the model
        :return:
    """
    model = create_model()
    if train is None and validation is None:
        train, validation, test = dataset.load_files(_dataset_dir)

    callbacks_list = [ModelCheckpoint(_model, monitor='val_loss', save_best_only=True),
                      TrainValTensorBoard(write_graph=False)]

    model.fit_generator(
        load_dataset(train, _dataset_dir, _batch_size),
        steps_per_epoch=steps(train, _batch_size), epochs=_epochs,
        validation_data=load_dataset(validation, _dataset_dir, _batch_size),
        validation_steps=steps(validation, _batch_size),
        callbacks=callbacks_list)


def predict_model(model=None, test=None):
    """
        Predict model
        :return:
    """
    x, y = load_sequence(test, _dataset_dir)
    y = model.predict(x)

    return y
