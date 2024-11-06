# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:28:38 2024

@author: robiul
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate sample data
time_cycle = np.arange(100)  # x-axis: Time cycle from 0 to 99
mean_y_predictions = np.random.rand(100)  # Mean y predictions
mean_x_predictions = np.random.rand(100)  # Mean x predictions
std_y_predictions = 0.1 * np.random.rand(100)  # Std of y predictions
std_x_predictions = 0.1 * np.random.rand(100)  # Std of x predictions

# Calculate upper and lower bounds for y and x
y_upper = mean_y_predictions + std_y_predictions
y_lower = mean_y_predictions - std_y_predictions
x_upper = mean_x_predictions + std_x_predictions
x_lower = mean_x_predictions - std_x_predictions

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot mean line
ax.plot(time_cycle, mean_y_predictions, mean_x_predictions, color='blue', label='Mean Prediction')

# Create mesh grids for upper and lower bounds
time_grid = np.tile(time_cycle, (2, 1))  # Repeat time_cycle for each bound

# Plot uncertainty surfaces for y predictions
ax.plot_surface(time_grid, np.vstack([y_lower, y_upper]), np.tile(mean_x_predictions, (2, 1)),
                color='lightblue', alpha=0.3, rstride=1, cstride=1, edgecolor='none')

# Plot uncertainty surfaces for x predictions
ax.plot_surface(time_grid, np.tile(mean_y_predictions, (2, 1)), np.vstack([x_lower, x_upper]),
                color='lightgreen', alpha=0.3, rstride=1, cstride=1, edgecolor='none')

# Set axis labels
ax.set_xlabel('Time Cycle')
ax.set_ylabel('Y Predictions')
ax.set_zlabel('X Predictions')

# Add legend
ax.legend()

# Display plot
plt.show()


