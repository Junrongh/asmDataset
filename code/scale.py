import os
import glob
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.filters import gaussian
# import matplotlib.pyplot as plt

from collections import defaultdict


def bot_crop(image):
    # Since the top & bottom have the scales and the marks,
    # which is useless for the future image representation,
    # they are cut in this part. The shape is usually around 75 pixels#
    x, y = np.array(image.shape).astype(int)
    imaged = image[0:x - 75, :]

    return imaged


def getBackgroundColor(im, edge=None):
    if edge is None:
        edge = 5

    x, y = im.shape

    top = im[:edge, :]
    bottom = im[x - edge:, :]
    left = im[:, :edge]
    right = im[:, y - edge:]

    colors = defaultdict(lambda: 0)
    for img in (left, right, top, bottom):
        w, h = img.shape
        for i in range(w):
            for j in range(h):
                c = img[i, j]
                colors[c] += 1

    colors_reversed = [(count, color) for (color, count) in colors.items()]
    background_color = max(colors_reversed)[1]
    return background_color


def imgshape(image):
    x, y = image.shape
    (estimatex, estimatey) = (int(x * 0.2), int(y * 0.2))
    average = np.average(image)
    estimate1 = image[:estimatex, :estimatey]
    estimate2 = image[x - estimatex:, :estimatey]
    estimate3 = image[:estimatex, y - estimatey:]
    estimate4 = image[x - estimatex:, y - estimatey:]
    estimateaverage = np.average(
        estimate1 + estimate2 + estimate3 + estimate4) / 4.0
    # 1 = white, 0 = black#
    if (1 - estimateaverage) < 0.3 * (1 - average):
        return "Circle"
    else:
        return "Rectangular"


def autocrop(image, threshold, imgshape, heldout, bgc, error_rate):
    x, y = image.shape
    top = int(heldout * x)
    bottom = int(x - int(heldout * x))
    left = int(heldout * y)
    right = int(y - heldout * y)
    edge = []
    for start, end, step in ((top, 0, -1), (bottom, x, 1)):
        for i in range(start, end, step):
            count = 0
            for j in image[i, :]:
                if j not in range(int(bgc - threshold), int(bgc + threshold)):
                    count = count + 1
            error = count * 1.0 / y
            if error < error_rate:
                break
        edge.append(i)

    for start, end, step in ((left, 0, -1), (right, y, 1)):
        for i in range(start, end, step):
            count = 0
            for j in image[:, i]:
                if j not in range(int(bgc - threshold), int(bgc + threshold)):
                    count = count + 1
            error = count * 1.0 / x
            if error < error_rate:
                break
        edge.append(i)
    if imgshape == "Rectangular":
        auto_crop = image[edge[0]:edge[1], edge[2]:edge[3]]

    else:
        centerx = int((edge[0] + edge[1]) / 2.0)
        centery = int((edge[2] + edge[3]) / 2.0)
        dltx = int((edge[1] - edge[0]) / (2.0 * 1.8))
        dlty = int((edge[3] - edge[2]) / (2.0 * 1.5))

        auto_crop = image[centerx - dltx:centerx +
                          dltx, centery - dlty:centery + dlty]
    return auto_crop


if __name__ == '__main__':

    # Showing Overview for testimg
    def overview():
        n = 4
        paths = glob.glob(os.path.join('testimg', '*.*'))
        fig, axes = plt.subplots(int((len(paths) - 1) / n) * 2 + 2, n, figsize=(32,72))
        for a in axes:
            for b in a:
               b.axis('off')
        i = 0
        for path in paths:
            im = np.array(imread(path, as_grey=True), dtype=float)

            im = im * 1.0 / np.max(im)
            bgc = getBackgroundColor(im, edge=100)
            im_b = bot_crop(im)
            auto_cropimg = autocrop(im_b, threshold=15, imgshape=imgshape(im_b), heldout=0.2,
                                    bgc=bgc, error_rate=0.1)
            j = int(i / n) * 2
            k = i % n
            axes[j, k].imshow(im, cmap='gray')
            axes[j+1, k].imshow(auto_cropimg, cmap='gray')
            i = i + 1
        # plt.show()
        plt.savefig('overview.png')

        return 0

    # Scaling for every img in the Micrographs floder


    def scaling(main_mt):
        paths = glob.glob(os.path.join('../Micrographs', main_mt, '*.*'))
        for path in paths:
            (floder, filename) = os.path.split(path)
            print('Formatting {}'.format(filename))
            im = np.array(imread(path, as_grey=True), dtype=float)
            bgc = getBackgroundColor(im, edge=100)
            im = bot_crop(im)
            auto_cropimg = autocrop(im, threshold=15, imgshape=imgshape(im), heldout=0.2,
                                    bgc=bgc, error_rate=0.1)
            scaled = resize(auto_cropimg, [224, 224])
            colored = gray2rgb(scaled).astype(np.float32)
            colored = colored/np.max(colored)
            imsave(os.path.join('../Micrographs_scaled', main_mt, filename), colored)

    scaling('al')
    scaling('as')
    scaling('cc')
    scaling('ci')
    scaling('co')
    scaling('cs')
    scaling('cu')
    scaling('hs')
    scaling('lz')
    scaling('mg')
    scaling('ni')
    scaling('pl')
    scaling('rf')
    scaling('sc')
    scaling('sp')
    scaling('ss')
    scaling('ti')
    scaling('ts')
    scaling('un')
