## This is course material for Introduction to Python Scientific Programming
## Example code: least_squares_penalty.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)

# subplot 1 plots the noisy samples
# Define the model
x = np.arange(-5, 5, 0.1)
y = 2*x - 1
plt.plot(x, y, 'gray', linewidth = 3)

# Generate noisy sample
sample_count = 100
x_sample = 10*np.random.random(sample_count)-5 # x_sample is generated randomly within the range [−5,5].
y_sample = 2*x_sample - 1 + np.random.normal(0, 1.0, sample_count) # y_sample is generated based on the model plus random noise (np.random.normal).

# Try some colormaps: hsv, gray, pink, cool, hot
ax1.scatter(x_sample, y_sample, c = x_sample, cmap = 'hsv')
# c = x_sample --> Specifies the color for each point based on its x-coordinate value, according to the color map 

# subplot 2 plots the parameter space
ax2 = fig.add_subplot(1,2,2, projection = '3d')

def penalty(para_a, para_b): # LLS penalty equation
    squares = (y_sample - para_a*x_sample - para_b)**2
    return 1/2/sample_count*np.sum(squares)

a_arr, b_arr = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1) )
# grids of parameter values over which the penalty function will be evaluated.

func_value = np.zeros(a_arr.shape) # make array initialized to 0s
for ax in range(a_arr.shape[0]): # a_arr.shape[0] --> num of rows in a_arr
    for ay in range(a_arr.shape[1]):#  --> num of columns in a_arr
            func_value[ax, ay] = penalty(a_arr[ax, ay], b_arr[ax, ay])

ax2.plot_surface(a_arr, b_arr, func_value, color = 'red', alpha = 0.8) # plot a surface
# alpha is transparency
ax2.set_xlabel('a parameter')
ax2.set_ylabel('b parameter')
ax2.set_zlabel('f(a, b)') 

# Find the minimum value
optimal_x, optimal_y = np.where(func_value == np.amin(func_value)) # elementwise 
print(a_arr[optimal_x, optimal_y], b_arr[optimal_x, optimal_y]) # prints best a and b
ax2.scatter(a_arr[optimal_x, optimal_y], b_arr[optimal_x, optimal_y], \
    func_value[optimal_x, optimal_y], marker = '*') # z coordinate
plt.show()