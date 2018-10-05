from os.path import isfile, join
from os import listdir
import matplotlib.pyplot as plt
from utilities import labels as lb
import os
import tqdm
import wfdb
import cv2
import json
import random

_range_to_ignore = 20
_directory = 'mitbih/'
_dataset_dir = 'dataset_no_augmentation/'
_split_percentage = .70


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
        For each beat for each patient creates img 128x128
        :param files:
    """
    if not os.path.exists(_directory):
        os.makedirs(_directory)

    files = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]

    random.shuffle(files)
    train = files[: int(len(files) * _split_percentage)]
    test = files[int(len(files) * _split_percentage):]
    print('TRAIN:\n', train, '\nTEST\n', test)

    for file in files:
        print('[INFO] START TO CONVERT FILE {}'.format((str(file))))
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
        print('\n[INFO] FILE {} IS CONVERTED'.format((str(file))))
