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