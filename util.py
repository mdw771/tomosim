# -*- coding: utf-8 -*-

import numpy as np
import tomopy
from scipy.ndimage import zoom, gaussian_filter

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


def trim_sinogram(data, center, x, y, diameter, fin_angle=180):
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

    delphi = np.pi * (fin_angle / 180) / dx
    for m in range(dx):

        # Calculate start end coordinates for each row of the sinogram.
        ind1 = np.ceil(np.cos(alpha - m * delphi) * (l2 - l1) + l1)
        ind2 = np.ceil(np.cos(alpha - m * delphi) * (l2 - l1) + l1 + diameter)

        # Make sure everythin is inside the frame.
        if ind1 < 0:
            ind1 = 0
        if ind1 > dz:
            ind1 = dz
        if ind2 < 0:
            ind2 = 0
        if ind2 > dz:
            ind2 = dz

        ind1, ind2 = map(int, [ind1, ind2])
        roidata[m, :, 0:(ind2 - ind1)] = data[m:m+1, :, ind1:ind2]
        mask[m, ind1:ind2] = True

    return roidata, mask


def normalize(img):

    return (img - img.min()) / (img.max() - img.min())


def snr(img, ref, mask_ratio=None):
    """
    Calculate the signal-to-noise ratio.
    :param img: image array
    :param ref: ground truth
    :return:
    """
    r = np.copy(ref)
    i = np.copy(img)
    if mask_ratio is None:
        rmean = r.mean()
        imean = i.mean()
        r = r + imean - rmean
    else:
        mask = tomopy.misc.corr._get_mask(img.shape[0], img.shape[1], mask_ratio)
        r = r[mask]
        i = i[mask]
        rmean = r.mean()
        imean = i.mean()
        r = r + imean - rmean
    return 10 * np.log10(np.linalg.norm(r) ** 2 / np.linalg.norm(r - i) ** 2)


def downsample_img(img, ds, axis=0):

    if isinstance(ds, int):
        if img.ndim == 3:
            res = tomopy.downsample(img, level=int(np.log2(ds)), axis=0)
        else:
            res = tomopy.downsample(img[:, np.newaxis, :], level=int(np.log2(ds)), axis=0)
    else:
        zm = np.ones(img.ndim)
        zm[axis] = 1. / ds
        res = zoom(img, zm)
    return res


def lateral_damp(img, length=50, sigma=None):

    ind = int(length / 2)
    mask = np.ones(img.shape)
    mask[:, :ind] = 0
    mask[:, -ind:] = 0
    if sigma is None:
        sigma = ind / 5.
    mask = gaussian_filter(mask, sigma=sigma)
    return img * mask

def pad_sinogram(sino, length):

    length = int(length)
    res = np.zeros([sino.shape[0], sino.shape[1] + length * 2])
    res[:, length:length+sino.shape[1]] = sino
    mean_left = np.mean(sino[:, :40], axis=1).reshape([sino.shape[0], 1])
    mean_right = np.mean(sino[:, -40:], axis=1).reshape([sino.shape[0], 1])
    res[:, :length] = mean_left
    res[:, -length:] = mean_right

    return res

