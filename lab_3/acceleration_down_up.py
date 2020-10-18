#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:42:40 2020

@author: xrachelpeng
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

# import data into arrays 
down = np.loadtxt("19a-InclinePlane-downward.txt", delimiter= ","). transpose()
