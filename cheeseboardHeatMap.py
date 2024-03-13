# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:11:40 2023

@author: Yang
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle
# Read the data from the sheet (assuming it's in a CSV file)
data = pd.read_csv('G:/JY/Jinghua_Sarah_chemo_BonsaiTracks/24hourProbeExp/932/932_S6_24HProbe/932_probe_adjust0.csv')  # Replace 'data.csv' with the actual file name

# Drop rows with NaN values
data = data.dropna(subset=['X', 'Y'])
data = data[data['X'] >=90] # Delete the bridge part.

# Extract the X and Y positions from the data
x = data['X']
y = data['Y']

# Create a 2D histogram using seaborn
heatmap, xedges, yedges = np.histogram2d(x, y, bins=20)  # Adjust the 'bins' parameter as needed
# Rotate the heatmap 90 degrees clockwise
heatmap = np.rot90(heatmap)
# Plot the heatmap with a blueish colormap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(heatmap, cmap='Blues')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Heatmap')

# Get the size of each bin in the heatmap
bin_size_x = (xedges[-1] - xedges[0]) / len(xedges)
bin_size_y = (yedges[-1] - yedges[0]) / len(yedges)

# Calculate the radius of the circle as half of the minimum bin size
circle_radius = min(bin_size_x, bin_size_y) / 2

# Calculate the center of the circle
center_x = (xedges[-1] + xedges[0]) / 2
center_y = (yedges[-1] + yedges[0]) / 2

# Plot the circle
circle = plt.Circle((center_x, center_y), radius=circle_radius, color='red', fill=False, linewidth=2)
ax.add_patch(circle)

# Invert the y-axis
ax.invert_yaxis()

plt.show()