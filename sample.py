# -*- coding: utf-8 -*-

import numpy as np
from constants import *
try:
    from xraylib import *
    lib_imp = True
except:
    lib_imp = False

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
        if lib_imp:
            res = CS_Total_CP(self.formula, energy) * self.density * 1e-4
        else:
            res = Mu * Rho * 1e-4

        return res