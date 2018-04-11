# -*- coding: utf-8 -*-

import numpy as np
import tomopy
from scipy.ndimage import zoom, gaussian_filter
# from scipy.stats import signaltonoise

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


def ssim(x, y, terms='lcs', mask_ratio=None):

    ssim = 1
    mask = tomopy.misc.corr._get_mask(x.shape[0], x.shape[1], mask_ratio)
    x = x[mask]
    y = y[mask]
    for term in terms:
        if term == 'l':
            mu_x = np.mean(x)
            mu_y = np.mean(y)
            c1 = (0.01 * (np.max([x.max(), y.max()]) - np.min([x.min(), y.min()])))**2
            ssim *= ((2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1))
        if term == 'c':
            sigma_x = np.sqrt(np.var(x))
            sigma_y = np.sqrt(np.var(y))
            c2 = (0.03 * (np.max([x.max(), y.max()]) - np.min([x.min(), y.min()])))**2
            ssim *= ((2 * sigma_x * sigma_y + c2) / (sigma_x**2 + sigma_y**2 + c2))**2
        if term == 's':
            t = np.vstack([x.flatten(), y.flatten()])
            t = np.cov(t)
            sigma_x = np.sqrt(t[0, 0])
            sigma_y = np.sqrt(t[1, 1])
            sigma_xy = t[0, 1]
            c3 = (0.03 * (np.max([x.max(), y.max()]) - np.min([x.min(), y.min()])))**2 / 2
            ssim *= ((sigma_xy + c3) / (sigma_x * sigma_y + c3))
    return ssim


def snr(img, ref, mask_ratio=None, ss_error=False):
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
        i = i + rmean - imean
    else:
        mask = tomopy.misc.corr._get_mask(img.shape[0], img.shape[1], mask_ratio)
        r = r[mask]
        i = i[mask]
        rmean = r.mean()
        imean = i.mean()
        i = i + rmean - imean
    if ss_error:
        return np.linalg.norm(r - i) ** 2
    else:
        return 10 * np.log10(np.linalg.norm(r) ** 2 / np.linalg.norm(r - i) ** 2)


def snr_intrinsic(img, mask_ratio=None):
    """
    Calculate the signal-to-noise ratio.
    :param img: image array
    :param ref: ground truth
    :return:
    """
    i = np.copy(img)
    if mask_ratio is not None:
        mask = tomopy.misc.corr._get_mask(img.shape[0], img.shape[1], mask_ratio)
        i = i[mask]
    snr = signaltonoise(i)
    return snr


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


def get_fov(n_scan, scanned_sino_width, mask_ratio_local):
    """
    Get field of view width that allows full coverage in local tomography.
    """
    fov = scanned_sino_width if n_scan == 1 else int(np.sqrt(2) * scanned_sino_width / (n_scan - 1 + np.sqrt(2))) + 10
    fov = int(fov / mask_ratio_local)
    if fov % 2 == 1:
        fov += 1
    return fov


def get_nscan_ps(f, gammaps, l):

    if f >= l:
        return 1
    else:
        return int(np.ceil((l - f) / (gammaps * f) + 1))


def get_nscan_os(f, fprime, l):

    if f >= l:
        return 1
    else:
        return int(np.ceil(np.sqrt(2) * l / fprime))