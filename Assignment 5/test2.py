# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:57:46 2024

@author: robiul
"""

import numpy as np

data = np.array([100, 200, 1, 2,  6, 7, 8, 9, 10, 3, 4, 5,])
q = np.percentile(data, 25)

data_x = np.random.rand(1000, 5, 99, 2)

out = np.percentile(data_x, 25, axis=1).reshape((-1, 1, 99, 2))
mask = (data_x>=out)
filtered_data = np.where(mask, data_x, np.nan)
output = np.nanpercentile(filtered_data, 75, axis=1).reshape((1000, 1, 99, 2))

a = np.array([[10,  7,  4],
       [ 3,  2,  1]])

lower_bound = np.percentile(a, 25, axis=0)
lower_bound = np.array([4.2, 1.5, 2])
a[(a>=lower_bound)]


lower_bound = np.percentile(data, 25)
upper_bound = np.percentile(data, 75)

data[(data >= lower_bound) & (data <= upper_bound)]

from scipy import stats
x, y = [1, 2, 3, 4, 5, 6, 7], [10, 9, 2.5, 6, 4, 3, 2]

s, p = stats.pearsonr(x, y)

