#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""Statistic abstract. """

import numpy as np

class ScalarContainer(object):
    def __init__(self):
        self.scalar_list = []
    def write(self, s):
        self.scalar_list.append(float(s))
    def read(self):
        ave = np.mean(np.array(self.scalar_list))
        self.scalar_list = []
        return ave

def count_consective_num(a, number):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, number).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


#a = [0, 1, 1]
#print (count_consective_num(a, 1))
#a = [0, 1,1, 0, 0]
#print (count_consective_num(a, 0))

