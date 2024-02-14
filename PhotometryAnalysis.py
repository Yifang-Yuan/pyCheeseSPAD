# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:13:41 2022
@author: Yifang
"""
import numpy as np
import matplotlib.pyplot as plt
import photometry_functions as fp
import scipy

# Folder with your files
#folder = 'C:/SPAD/pyPhotometry_v0.3.1/data/' # Modify it depending on where your file is located
folder ='E:/Qingren/1002/Day2/'
COLD_folder='E:/Qingren/'
# File name
file_name = '1002-2023-08-12-180348_9.csv'
sync_filename='1002_Sync_9.csv'
COLD_filename='Training Data_adjusted.xlsx'

sampling_rate=130
CamFs=24
#%%
raw_signal,raw_reference,Cam_Sync=fp.read_photometry_data (folder, file_name, readCamSync='True',plot=False)
#raw_signal,raw_reference=read_photometry_data (folder, file_name, readCamSync='False')
CamSync_LED=fp.read_Bonsai_Sync (folder, sync_filename,plot=False)
fp.plot_sync (raw_signal,raw_reference,Cam_Sync,CamSync_LED,pyFs=sampling_rate,CamFs=CamFs)
#%%
zdFF = fp.get_zdFF(raw_reference,raw_signal,smooth_win=5,remove=0,lambd=5e4,porder=1,itermax=50)
fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(111)
ax1 = fp.plotSingleTrace (ax1, zdFF, SamplingRate=sampling_rate,color='black',Label='zscore_signal')
#%%
zscore_sync,Sync_index_inCam,Sync_Start_time=fp.sync_photometry_Cam(zdFF,Cam_Sync,CamSync_LED,CamFs=CamFs)
reference_sync=fp.Cut_photometry_data (raw_reference, Cam_Sync)
signal_sync=fp.Cut_photometry_data (raw_signal, Cam_Sync)
#%%
'''
From here, you'll need COLD output to get the PETH plot
'''
cheeaseboard_session_data=fp.read_cheeseboard_from_COLD (COLD_folder, COLD_filename)
'''for this trial'''
entertime, well1time,well2time=fp.adjust_time_to_photometry(cheeaseboard_session_data,9,Sync_Start_time)
half_timewindow=2
fig = plt.figure(figsize=(16, 5))
ax = fig.add_subplot(111)
fp.PETH_plot_zscore(ax, zscore_sync,centre_time=well2time, half_timewindow=half_timewindow, fs=sampling_rate,color='black')
ax.axvline(x=2, color='red', linestyle='--', label='Event Time')

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(211)
fp.PETH_plot_zscore(ax1, signal_sync,centre_time=well2time, half_timewindow=half_timewindow, fs=sampling_rate,color='blue')
ax1.axvline(x=2, color='red', linestyle='--', label='Event Time')
ax2 = fig.add_subplot(212)
fp.PETH_plot_zscore(ax2, reference_sync,centre_time=well2time, half_timewindow=half_timewindow, fs=sampling_rate,color='purple')
ax2.axvline(x=2, color='red', linestyle='--', label='Event Time')
#%%
'''
You can get zdFF directly by calling the function fp.get_zdFF()
TO CHECK THE SIGNAL STEP BY STEP:
YOU CAN USE THE FOLLOWING CODES TO GET MORE PLOTS
These will give you plots for 
smoothed signal, corrected signal, normalised signal and the final zsocre
'''
smooth_win = 10
smooth_reference,smooth_signal,r_base,s_base = fp.photometry_smooth_plot (
    raw_reference,raw_signal,sampling_rate=sampling_rate, smooth_win = smooth_win)
#%%
remove=0
reference = (smooth_reference[remove:] - r_base[remove:])
signal = (smooth_signal[remove:] - s_base[remove:])  

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1 = fp.plotSingleTrace (ax1, signal, SamplingRate=sampling_rate,color='blue',Label='corrected_signal')
ax2 = fig.add_subplot(212)
ax2 = fp.plotSingleTrace (ax2, reference, SamplingRate=sampling_rate,color='purple',Label='corrected_reference')

#%%
z_reference = (reference - np.median(reference)) / np.std(reference)
z_signal = (signal - np.median(signal)) / np.std(signal)

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1 = fp.plotSingleTrace (ax1, z_signal, SamplingRate=sampling_rate,color='blue',Label='normalised_signal')
ax2 = fig.add_subplot(212)
ax2 = fp.plotSingleTrace (ax2, z_reference, SamplingRate=sampling_rate,color='purple',Label='normalised_reference')

#%%
from sklearn.linear_model import Lasso
lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
n = len(z_reference)

lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))

z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(111)
ax1 = fp.plotSingleTrace (ax1, z_signal, SamplingRate=sampling_rate,color='blue',Label='normalised_signal')
ax1 = fp.plotSingleTrace (ax1, z_reference_fitted, SamplingRate=sampling_rate,color='purple',Label='fitted_reference')
#%%
zdFF = (z_signal - z_reference_fitted)
fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(111)
ax1 = fp.plotSingleTrace (ax1, zdFF[10*sampling_rate:20*sampling_rate], SamplingRate=sampling_rate,color='black',Label='zscore_signal')

#%%
from scipy import signal
f, t, Sxx = signal.spectrogram(raw_signal, fs=130)
plt.pcolormesh(t, f, Sxx, shading='gouraud',vmax=800)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 20)
plt.show()
