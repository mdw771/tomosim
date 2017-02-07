# -*- coding: utf-8 -*-

import numpy as np


class Instrument(object):

    def __init__(self, field_of_view, resolution):
        """
        :param field_of_view: width of field of view in pixel
        :param resolution: real space resolution in um
        :return:
        """
        self.fov = int(field_of_view)
        self.resolution = resolution
        self.stage_positions = []
        self.center_positions = []

    def add_stage_positions(self, stage_pos):
        """
        Takes a uniaxial coordination or a list of coordinations.
        :param stage_pos: pixel positions of sample stage
        :return:
        """
        if isinstance(stage_pos, np.ndarray):
            stage_pos = stage_pos.tolist()
        if not isinstance(stage_pos, list):
            self.stage_positions.append(stage_pos)
        else:
            self.stage_positions += stage_pos

    def add_center_positions(self, center_pos):
        """
        Takes a coordiantion tuple or a list of tuples.
        :param center_pos: pixel positions of rotation centers based on sample grid at 0 deg rotation
        :return:
        """
        if isinstance(center_pos, np.ndarray):
            center_pos = center_pos.tolist()
        if isinstance(center_pos[0], list) or isinstance(center_pos[0], tuple):
            for i in center_pos:
                self.center_positions.append(tuple(i))
        else:
            self.center_positions.append(tuple(center_pos))

