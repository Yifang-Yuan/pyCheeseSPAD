# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:09:09 2024

@author: Yifang
"""
'To batch process multiple days and save event_window_traces'
import matplotlib.pyplot as plt
import pyCheeseSession
import os
import pandas as pd
from scipy import stats
import numpy as np


def Plot_mean_With_CI_PSTH_segment(event_window_traces, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=None):
    event_window_traces = event_window_traces.dropna(axis=1, how='all')
    'find segment based on before windwo and after window'
    'This is different from the Plot_mean_With_CI_PSTH in photometry_functions'
    sampling_rate = 130  # in Hz
    # Calculate the midpoint of the data (1300 samples)
    midpoint = len(event_window_traces) // 2  # This gives the index of the midpoint, which is 650
    # Calculate the number of samples before and after the midpoint
    before_samples = before_window * sampling_rate  # 2 * 130 = 260 samples
    after_samples = after_window * sampling_rate    # 2 * 130 = 260 samples
    # Determine the start and end indices for the segment
    start_index = midpoint - before_samples  # 650 - 260 = 390
    end_index = midpoint + after_samples     # 650 + 260 = 910
    segment = event_window_traces.iloc[start_index:end_index]
    
    mean_signal = segment.mean(axis=1)
    sem = stats.sem(segment, axis=1, nan_policy='omit')
    df = len(segment.columns) - 1
    moe = stats.t.ppf(0.975, df) * sem  # 0.975 for 95% confidence level (two-tailed)
    event_time = 0
    num_samples = len(mean_signal)
    time_in_seconds = np.linspace(-before_window, after_window, num_samples)

    # If an 'ax' is provided, use it for plotting; otherwise, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_in_seconds, mean_signal, label='Mean Signal', color=meancolor,linewidth=1)
    ax.fill_between(time_in_seconds, mean_signal - moe, mean_signal + moe, color=stdcolor, alpha=0.5, label='95% CI')
    #print (mean_signal - moe)
    ax.axvline(x=event_time, color='red', linestyle='--', label='Event Time')
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('zscore')
    ax.set_title('Mean Signal with CI ' + animalID)
    #ax.legend()

    # If 'ax' was provided, do not call plt.show() to allow the caller to display or save the figure as needed
    if ax is None:
        plt.show()
    return ax

def Plot_mean_PSTH_segment(event_window_traces, before_window, after_window, animalID, Label='Mean',ax=None):
    event_window_traces = event_window_traces.dropna(axis=1, how='all')
    'find segment based on before windwo and after window'
    'This is different from the Plot_mean_With_CI_PSTH in photometry_functions'
    sampling_rate = 130  # in Hz
    # Calculate the midpoint of the data (1300 samples)
    midpoint = len(event_window_traces) // 2  # This gives the index of the midpoint, which is 650
    # Calculate the number of samples before and after the midpoint
    before_samples = before_window * sampling_rate  # 2 * 130 = 260 samples
    after_samples = after_window * sampling_rate    # 2 * 130 = 260 samples
    # Determine the start and end indices for the segment
    start_index = midpoint - before_samples  # 650 - 260 = 390
    end_index = midpoint + after_samples     # 650 + 260 = 910
    segment = event_window_traces.iloc[start_index:end_index]
    
    mean_signal = segment.mean(axis=1)
    sem = stats.sem(segment, axis=1, nan_policy='omit')
    df = len(segment.columns) - 1
    moe = stats.t.ppf(0.975, df) * sem  # 0.975 for 95% confidence level (two-tailed)
    event_time = 0
    num_samples = len(mean_signal)
    time_in_seconds = np.linspace(-before_window, after_window, num_samples)

    # If an 'ax' is provided, use it for plotting; otherwise, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_in_seconds, mean_signal, label=Label, linewidth=1)
    #ax.fill_between(time_in_seconds, mean_signal - moe, mean_signal + moe, alpha=0.5, label='95% CI')
    #print (mean_signal - moe)
    ax.axvline(x=event_time, color='red', linestyle='--')
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('zscore')
    ax.set_title('Mean Signal with CI ' + animalID)
    ax.legend()
    # If 'ax' was provided, do not call plt.show() to allow the caller to display or save the figure as needed
    if ax is None:
        plt.show()
    return ax

def Concat_PETH_pkl_files (parent_folder, target_string='traces.pkl'):
    dfs = []
    files = os.listdir(parent_folder)
    filtered_files = [file for file in files if target_string in file]
    for filename in filtered_files:
        print (filename)
        # Read the DataFrame from the .pkl file
        df = pd.read_pickle(os.path.join(parent_folder, filename))            
        # Extract an index from the file name (you may need to customize this)
        day_index = filename.split('_')[1] # Assuming the file names are like "1.pkl", "2.pkl", etc.            
        # Rename the columns with the extracted index
        df.columns = [f'{day_index}_{col}' for col in df.columns]            
        # Append the DataFrame to the list
        dfs.append(df)   
        # Concatenate the DataFrames column-wise
        result_df = pd.concat(dfs, axis=1)
    return result_df

def plot_2wells_PETH_all_trials (result_folder,before_window,after_window,animalID):
    PSTH_collection=Concat_PETH_pkl_files (result_folder, target_string='win_traces')
    filtered_columns = [col for col in PSTH_collection.columns if col.endswith('_1')]
    Well1_PETH = PSTH_collection[filtered_columns]

    filtered_columns = [col for col in PSTH_collection.columns if col.endswith('_2')]
    Well2_PETH = PSTH_collection[filtered_columns]
    '''plot'''
    fig, ax = plt.subplots(figsize=(10, 4))
    Plot_mean_With_CI_PSTH_segment(Well1_PETH, before_window, after_window, animalID, meancolor='green', stdcolor='lightgreen', ax=ax)
    ax.set_title('Well1 PETH '+animalID)  # Set the title for the first figure
    ax.legend(['Mean GCamp8m', '95% CI'])  # Add a legend

    fig, ax = plt.subplots(figsize=(10, 4))
    Plot_mean_With_CI_PSTH_segment(Well2_PETH, before_window, after_window, animalID, meancolor='red', stdcolor='lightcoral', ax=ax)
    ax.set_title('Well2 PETH '+animalID)  # Set the title for the first figure
    ax.legend(['Mean GCamp8m', '95% CI'])  # Add a legend

    fig, ax = plt.subplots(figsize=(10, 4))
    Plot_mean_With_CI_PSTH_segment(PSTH_collection, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=ax)
    ax.set_title('Both well PETH '+animalID)  # Set the title for the first figure
    ax.legend(['Mean GCamp8m', '95% CI'])  # Add a legend
    return -1

def plot_day_average_PETH_together(result_folder,before_window,after_window,animalID):
    PSTH_collection=Concat_PETH_pkl_files (result_folder, target_string='win_traces')
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range (5):
        idx=i+1
        filtered_columns = list(filter(lambda col: col.startswith(f'Day{idx}') and col.endswith('_1'), PSTH_collection.columns))
        Day_Well_PETH = PSTH_collection[filtered_columns]
        Plot_mean_PSTH_segment(Day_Well_PETH, before_window, after_window, animalID, Label=f'Day{idx} Mean',ax=ax)
    ax.set_title('Well1 PETH over days '+animalID)  # Set the title for the first figure
    
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range (5):
        idx=i+1
        filtered_columns = list(filter(lambda col: col.startswith(f'Day{idx}') and col.endswith('_2'), PSTH_collection.columns))
        Day_Well_PETH = PSTH_collection[filtered_columns]
        Plot_mean_PSTH_segment(Day_Well_PETH, before_window, after_window, animalID,Label=f'Day{idx} Mean',ax=ax)
    ax.set_title('Well2 PETH over days' + animalID)  # Set the title for the first figure
    return -1

def plot_SB_PETH_all_trials (result_folder,before_window,after_window,animalID):
    PSTH_collection=Concat_PETH_pkl_files (result_folder, target_string='twosides_on_startbox')
    filtered_columns = [col for col in PSTH_collection.columns if col.endswith('_enter')]
    SB_PETH = PSTH_collection[filtered_columns]
    '''plot'''
    fig, ax = plt.subplots(figsize=(10, 4))
    Plot_mean_With_CI_PSTH_segment(SB_PETH, before_window, after_window, animalID, meancolor='grey', stdcolor='lightgrey', ax=ax)
    ax.set_title('Leaving Startbox and Enter Cheeseboard PETH '+animalID)  # Set the title for the first figure
    ax.legend(['Mean GCamp8m', '95% CI'])  # Add a legend
    return -1

def plot_day_average_SB_PETH_together(result_folder,before_window,after_window,animalID):
    PSTH_collection=Concat_PETH_pkl_files (result_folder, target_string='twosides_on_startbox')
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range (5):
        idx=i+1
        filtered_columns = list(filter(lambda col: col.startswith(f'Day{idx}') and col.endswith('_enter'), PSTH_collection.columns))
        Day_Well_PETH = PSTH_collection[filtered_columns]
        Plot_mean_PSTH_segment(Day_Well_PETH, before_window, after_window, animalID, Label=f'Day{idx} Mean',ax=ax)
    ax.set_title('Enter Cheeseboard PETH over days '+animalID)  # Set the title for the first figure
    return -1