import numpy as np


class Sinogram(object):

    def __init__(self, type, sinogram, coords):

        assert type in ('local', 'tomosaic')

        self.type = type
        self.sinogram = sinogram
        self.coords = coords