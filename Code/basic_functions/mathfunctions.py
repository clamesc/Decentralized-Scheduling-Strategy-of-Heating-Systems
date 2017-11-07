# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 08:51:04 2014

@author: tsz
"""

import numpy as np

def interpolation(x_set, x, y):
    """ Interpolate x,y and to compute y_set at a given x_set 
        The parameters are:
        x_set:  Set value for the interpolation
        x:      List/Array with two entries. The first is for the lower x-value, the second for the upper x-value
        y:      List/Array with two entries that correspond to the lower x-value and the higher x-value
    """
    
    y_set = y[0] + float(x_set - x[0]) / (x[1] - x[0]) * (y[1] - y[0])
    
    return y_set