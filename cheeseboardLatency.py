# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 23:43:02 2023

@author: Yang
"""

import numpy as np
import matplotlib.pyplot as plt

# Sample data (average trial latency)
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', '24H probe']
rewards = np.array([
    [105, 43, 18, 35, 15, 10],
    [11, 30, 16, 25, 12, np.nan],
    [37, 41, 8, 47, 11, np.nan],
    [11, 59, 7, 13, 11, np.nan],
    [6, 30, 10, 19, 12, np.nan],
    [79, 46, 8, 33, 18, np.nan],
    [17, 18, 13, 40, 16, np.nan],
    [19, 20, 48, 41, 11, np.nan],
    [64, 15, 12, 14, 15, np.nan],
    [80, 23, 10, 24, 16, np.nan]
])

# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

for i, day in enumerate(days[:-1]):
    x = np.repeat(i, rewards.shape[0])
    y = rewards[:, i].astype(float)
    plt.scatter(x, y, color='gray', alpha=0.7)

# Calculate and plot the average
average_latency = np.nanmean(rewards[:, :-1].astype(float), axis=0)
x_avg = np.arange(len(days) - 1)
plt.scatter(x_avg, average_latency, color='blue', label='Average')

# Add the new data point
x_new = np.repeat(len(days) - 1, rewards.shape[0])
y_new = rewards[:, -1].astype(float)
plt.scatter(x_new, y_new, color='red', label='24H probe-reward')

# Customize the plot
plt.title('Mouse 932 - Found reward Latency', fontsize=16)
plt.xlabel('Days', fontsize=16)
plt.ylabel('Latency (s)', fontsize=16)
plt.xticks(range(len(days)), days, fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, 125)

# Display the legend
plt.legend(fontsize=16)

# Display the plot
plt.show()

#%%
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', '24H probe']
rewards = np.array([
    [120, 147, 76, 47, 23, 43],
    [20, 49, 24, 21, 17, np.nan],
    [102, 44, 46, 21, 21, np.nan],
    [np.nan, 28, 24, 22, 13, np.nan],
    [66, 135, 20, 30, 16, np.nan],
    [np.nan, 44, 26, 14, 17, np.nan],
    [21, 37, 13, 34, 13, np.nan],
    [65, 54, 17, 13, 13, np.nan],
    [38, 26, 18, 25, 11, np.nan],
    [26, 81, 9, 25, 12, np.nan]
])


# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

for i, day in enumerate(days[:-1]):
    x = np.repeat(i, rewards.shape[0])
    y = rewards[:, i].astype(float)
    plt.scatter(x, y, color='gray', alpha=0.7)

# Calculate and plot the average
average_latency = np.nanmean(rewards[:, :-1].astype(float), axis=0)
x_avg = np.arange(len(days) - 1)
plt.scatter(x_avg, average_latency, color='blue', label='Average')

# Add the new data point
x_new = np.repeat(len(days) - 1, rewards.shape[0])
y_new = rewards[:, -1].astype(float)
plt.scatter(x_new, y_new, color='red', label='24H probe-reward')

# Customize the plot
plt.title('Mouse 940 - Found reward Latency', fontsize=16)
plt.xlabel('Days', fontsize=16)
plt.ylabel('Latency (s)', fontsize=16)
plt.xticks(range(len(days)), days, fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, 125)

# Display the legend
plt.legend(fontsize=16,loc='upper right')

# Display the plot
plt.show()
#%%
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', '24H probe']
rewards = np.array([
    [np.nan, 60, 37, 63, 74, 120],
    [47, 120, 55, 65, 82, np.nan],
    [np.nan, 33, 79, 44, 56, np.nan],
    [np.nan, 93, 65, 90, np.nan, np.nan],
    [np.nan, 96, 75, 59, 90, np.nan],
    [10, 92, 65, 54, 62, np.nan],
    [np.nan, 80, 85, 57, 35, np.nan],
    [np.nan, 41, 50, 88, 102, np.nan],
    [np.nan, 58, 131, 65, 60, np.nan],
    [np.nan, 80, 88, 87, 75, np.nan]
])


# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

for i, day in enumerate(days[:-1]):
    x = np.repeat(i, rewards.shape[0])
    y = rewards[:, i].astype(float)
    plt.scatter(x, y, color='gray', alpha=0.7)

# Calculate and plot the average
average_latency = np.nanmean(rewards[:, :-1].astype(float), axis=0)
x_avg = np.arange(len(days) - 1)
plt.scatter(x_avg, average_latency, color='blue', label='Average')

# Add the new data point
x_new = np.repeat(len(days) - 1, rewards.shape[0])
y_new = rewards[:, -1].astype(float)
plt.scatter(x_new, y_new, color='red', label='24H probe-reward')

# Customize the plot
plt.title('Mouse 941 - Found reward Latency', fontsize=16)
plt.xlabel('Days', fontsize=16)
plt.ylabel('Latency (s)', fontsize=16)
plt.xticks(range(len(days)), days, fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, 125)

# Display the legend
#plt.legend(fontsize=16,loc='upper right')

# Display the plot
plt.show()