# -*- coding: utf-8 -*-

import numpy as np
import gc
import operator


def arrange_image(img1, img2, shift):

    shift = map(int, shift)
    new_shape = map(int, map(max, map(operator.add, img2.shape, shift), img1.shape))
    newimg = np.empty(new_shape)
    newimg[:, :] = np.NaN
    newimg[0:img1.shape[0], 0:img1.shape[1]] = img1
    newimg[shift[0]:shift[0] + img2.shape[0], shift[1]:shift[1] + img2.shape[1]] = img2
    gc.collect()

    return newimg


def trim_sinogram(data, center, x, y, diameter):
    """
    Adopted from Tomopy.
    Provide sinogram corresponding to a circular region of interest
    by trimming the complete sinogram of a compact object.

    Parameters
    ----------
    data : ndarray
        Input 3D data.

    center : float
        Rotation center location.

    x, y : int, int
        x and y coordinates in pixels (image center is (0, 0))

    diameter : float
        Diameter of the circle of the region of interest.

    Returns
    -------
    ndarray
        Output 3D data.

    """
    data = data.copy().astype('float32')
    dx, dy, dz = data.shape
    mask = np.zeros([dx, dz], dtype='bool')

    rad = np.sqrt(x * x + y * y)
    alpha = np.arctan2(x, y)
    l1 = center - diameter / 2
    l2 = center - diameter / 2 + rad

    roidata = np.ones((dx, dy, diameter), dtype='float32')

    delphi = np.pi / dx
    for m in range(dx):

        # Calculate start end coordinates for each row of the sinogram.
        ind1 = np.ceil(np.cos(alpha - m * delphi) * (l2 - l1) + l1)
        ind2 = np.floor(np.cos(alpha - m * delphi) * (l2 - l1) + l1 + diameter)

        # Make sure everythin is inside the frame.
        if ind1 < 0:
            ind1 = 0
        if ind1 > dz:
            ind1 = dz
        if ind2 < 0:
            ind2 = 0
        if ind2 > dz:
            ind2 = dz

        roidata[m, :, 0:(ind2 - ind1)] = data[m:m+1, :, ind1:ind2]
        mask[m, ind1:ind2] = True

    return roidata, mask


def snr(img, ref):
    """
    Calculate the signal-to-noise ratio.
    :param img: image array
    :param ref: ground truth
    :return:
    """
    return 10 * np.log10(np.linalg.norm(ref) ** 2 / np.linalg.norm(ref - img) ** 2)