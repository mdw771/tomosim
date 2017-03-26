# -*- coding: utf-8 -*-

import numpy as np
from xraylib import *

class Sample(object):

    def __init__(self, formula, density):
        """
        :param formula: chemical formula
        :param density: density in g/cm^3
        :return:
        """

        self.formula = formula
        self.density = density

    def get_attenuation_coeff(self, energy):
        """
        Linear attenuation coefficient
        :param energy: energy in keV
        :return: linear attenuation coefficient in um^-1
        """

        return CS_Total_CP(self.formula, energy) * self.density * 1e-4