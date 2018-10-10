import matplotlib.pyplot as plt
import tqdm
import random
import cv2
from os.path import isfile, join
from os import listdir
from utilities import labels as lb
import os
import wfdb
import numpy as np

_range_to_ignore = 20
_directory = '../Data/mitbih/'
_dataset_dir = '../Data/dataset_filtered/'
_dataset_ann_dir = '../Data/dataset_ann/'
_split_percentage = .70
_split_validation_percentage = 0.70
_split_test_percentage = 0.50
_width = 2503
_height = 3361


def cropping(image, filename, size):
    """
        :param image: the image to crop
        :param filename:
        :param size: prefered size
        :return:
    """
    # Left Top Crop
    crop = image[:96, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '1' + '.png', crop)

    # Center Top Crop
    crop = image[:96, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '2' + '.png', crop)

    # Right Top Crop
    crop = image[:96, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '3' + '.png', crop)

    # Left Center Crop
    crop = image[16:112, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '4' + '.png', crop)

    # Center Center Crop
    crop = image[16:112, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '5' + '.png', crop)

    # Right Center Crop
    crop = image[16:112, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '6' + '.png', crop)

    # Left Bottom Crop
    crop = image[32:, :96]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '7' + '.png', crop)

    # Center Bottom Crop
    crop = image[32:, 16:112]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '8' + '.png', crop)

    # Right Bottom Crop
    crop = image[32:, 32:]
    crop = cv2.resize(crop, size)
    cv2.imwrite(filename[:-5] + '9' + '.png', crop)


def create_img_from_sign(size=(128, 128), augmentation=True):
    """
       For each beat for each patient creates img apply some filters
       :param size: the img size
       :param augmentation: create for each image another nine for each side
    """
    if not os.path.exists(_directory):
        os.makedirs(_directory)

    files = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]

    random.shuffle(files)
    train = files[: int(len(files) * _split_percentage)]
    test = files[int(len(files) * _split_percentage):]

    for file in files:
        sig, _ = wfdb.rdsamp(_directory + file)
        ann = wfdb.rdann(_directory + file, extension='atr')
        for i in tqdm.tqdm(range(1, len(ann.sample) - 1)):

            if ann.symbol[i] not in lb.original_labels:
                continue
            label = lb.original_labels[ann.symbol[i]]
            if file in train:
                dir = '{}train/{}'.format(_dataset_dir, label)
            else:
                dir = '{}validation/{}'.format(_dataset_dir, label)
            if not os.path.exists(dir):
                os.makedirs(dir)

            ''' Get the Q-peak intervall '''
            start = ann.sample[i - 1] + _range_to_ignore
            end = ann.sample[i + 1] - _range_to_ignore

            ''' Get the signals '''
            plot_x = [sig[i][0] for i in range(start, end)]
            plot_y = [i * 1 for i in range(start, end)]

            ''' Plot and save the beat'''
            fig = plt.figure(frameon=False)
            plt.plot(plot_y, plot_x)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            ''' Convert in gray scale and resize img '''
            if file in train:
                filename = '{}train/{}/{}_{}{}{}0.png'.format(_dataset_dir, label, label, file[-3:], start, end)
            else:
                filename = '{}validation/{}/{}_{}{}{}0.png'.format(_dataset_dir, label, label, file[-3:], start, end)
            fig.savefig(filename)
            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, size, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, im_gray)
            if augmentation:
                cropping(im_gray, filename, size)
            plt.cla()
            plt.clf()
            plt.close('all')


def create_img_from_sign_filtered(size=(128, 128), size_paa=100, augmentation=True):
    """
        For each beat for each patient creates img apply some filters
        :param size: the img size
        :param augmentation: create for each image another nine for each side
    """
    if not os.path.exists(_directory):
        os.makedirs(_directory)

    files = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]

    for file in files:
        sig, _ = wfdb.rdsamp(_directory + file)
        ann = wfdb.rdann(_directory + file, extension='atr')
        for i in tqdm.tqdm(range(1, len(ann.sample) - 1)):

            if ann.symbol[i] not in lb.original_labels:
                continue
            label = lb.original_labels[ann.symbol[i]]
            ''' Get the Q-peak intervall '''
            start = ann.sample[i - 1] + _range_to_ignore
            end = ann.sample[i + 1] - _range_to_ignore

            signal = [sig[i][0] for i in range(start, end)]
            paa = piecewise_aggregate_approximation(signal, size_paa)
            plot_x = paa
            plot_y = [i for i in range(len(paa))]
            ''' Plot and save the beat'''
            fig = plt.figure(frameon=False)
            plt.plot(plot_y, plot_x)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            filename = '{}{}_{}{}{}0.png'.format(_dataset_dir, label, file[-3:], start, end)
            fig.savefig(filename, bbox_inches='tight')
            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, size, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, im_gray)
            plt.cla()
            plt.clf()
            plt.close('all')


def create_json_from_sign_filtered(size_paa=100):
    """
        For each beat for each patient creates img apply some filters
        :param size_paa: the new vector size
    """
    if not os.path.exists(_directory):
        os.makedirs(_directory)

    files = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]

    for f in tqdm.tqdm(range(len(files))):
        file = files[f]
        sig, _ = wfdb.rdsamp(_directory + file)
        ann = wfdb.rdann(_directory + file, extension='atr')
        for i in range(1, len(ann.sample) - 1):

            if ann.symbol[i] not in lb.original_labels:
                continue
            label = lb.original_labels[ann.symbol[i]]

            ''' Get the Q-peak intervall '''
            start = ann.sample[i - 1] + _range_to_ignore
            end = ann.sample[i + 1] - _range_to_ignore

            signal = [sig[i][0] for i in range(start, end)]
            paa = piecewise_aggregate_approximation(signal, size_paa)
            filename = '{}{}_{}{}{}.txt'.format(_dataset_ann_dir, label, file[-3:], start, end)
            with open(filename, 'w+') as f:
                for i, item in enumerate(paa):
                    f.write("%s" % item)
                    if i < len(paa) - 1:
                        f.write(", ")


def piecewise_aggregate_approximation(vector, paa_dim: int):
    '''
        Transform signals in a vector of size M
        :param vector: signals
        :param paa_dim: the new size
        :return:
    '''

    Y = np.array(vector)
    if Y.shape[0] % paa_dim == 0:
        sectionedArr = np.array_split(Y, paa_dim)
        res = np.array([item.mean() for item in sectionedArr])
    else:
        value_space = np.arange(0, Y.shape[0] * paa_dim)
        output_index = value_space // Y.shape[0]
        input_index = value_space // paa_dim
        uniques, nUniques = np.unique(output_index, return_counts=True)
        res = [Y[indices].sum() / Y.shape[0] for indices in
               np.split(input_index, nUniques.cumsum())[:-1]]
    return res


def load_files(directory):
    """
        Load each name file in the directory
        :param directory:
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
