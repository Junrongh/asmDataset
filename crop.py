#!/usr/bin/env python

import os
import glob
import numpy as np
from skimage.io import imread, imsave


def topbot_crop(image):
    # Since the top & bottom have the scales and the marks,
    # which is useless for the future image representation,
    #they are cut in this part. The shape is usually around 75 pixels#
    x, y, *rest = np.array(image.shape).astype(int)
    imaged = image[75:x - 75, 0:y - 1]

    return imaged


def selfcrop(image):
    x, y, *rest = np.array(image.shape).astype(float)
    (estimatex, estimatey) = (x * 0.2, y * 0.2)
    estimatex = int(estimatex)
    estimatey = int(estimatey)
    average = np.average(image)
    estimate = image[0:estimatex, 0:estimatey]
    estimateaverage = np.average(estimate)
    #255 means white, 0 means black#
    if (255 - estimateaverage) < 0.3 * (255 - average):
        return circle_crop(image)
    else:
        return quad_crop(image)


def quad_crop(image):
    x, y, *rest = np.array(image.shape).astype(int)
    if x > 550:
        image = topbot_crop(image)
    x, y, *rest = np.array(image.shape).astype(int)
    (centerx, centery) = (x / 2, y / 2)
    n = centerx / 200
    m = centery / 200
    centerx = int(centerx)
    centery = int(centery)
    i = 0
    crops = []
    while i < n - 1:
        j = 0
        while j < m - 1:
            crops.append((i, j, 1, image[
                         centerx - 200 * (i + 1):centerx - 200 * i, centery - 200 * (j + 1):centery - 200 * j]))
            crops.append((i, j, 2, image[
                         centerx - 200 * (i + 1):centerx - 200 * i, centery + 200 * j:centery + 200 * (j + 1)]))
            crops.append((i, j, 3, image[
                         centerx + 200 * i:centerx + 200 * (i + 1), centery - 200 * (j + 1):centery - 200 * j]))
            crops.append((i, j, 4, image[
                         centerx + 200 * i:centerx + 200 * (i + 1), centery + 200 * j:centery + 200 * (j + 1)]))
            j = j + 1
        i = i + 1

    return crops


def circle_crop(image):
    #choosing number 400 intent to crop 4 different parts of the image#
    x, y, *rest = np.array(image.shape).astype(int)
    (centerx, centery) = (x / 2, y / 2)
    centerx = int(centerx)
    centery = int(centery)
    image = image[centerx - 400:centerx + 400, centery - 400:centery + 400]
    return quad_crop(image)


if __name__ == '__main__':
    paths = glob.glob('Micrographs/*/*.png')

    for path in paths:
        print('cropping {}'.format(path))
        prefix, ext = os.path.splitext(os.path.basename(path))
        crops = selfcrop(imread(path, as_grey=True))
        for i, j, label, crop in crops:
            dest = '{}-crop{}_{}_{}.png'.format(prefix, i, j, label)
            dest = os.path.join('crops', dest)
            imsave(dest, crop)
