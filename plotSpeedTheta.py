# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:14:10 2024

@author: Yifang
"""

import pandas as pd
from waveletFunctions import wavelet
import os
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import filtfilt
from scipy import signal
import numpy as np
import seaborn as sns

def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=5): 
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def band_pass_filter(data,low_freq,high_freq,Fs):
    data_high=butter_filter(data, btype='high', cutoff=low_freq,fs=Fs, order=5)
    data_low=butter_filter(data_high, btype='low', cutoff=high_freq, fs=Fs, order=5)
    return data_low

def Calculate_wavelet(signal_pd,lowpassCutoff=1500,Fs=10000,scale=40):
    if isinstance(signal_pd, np.ndarray)==False:
        signal=signal_pd.to_numpy()
    else:
        signal=signal_pd
    sst = butter_filter(signal, btype='low', cutoff=lowpassCutoff, fs=Fs, order=5)
    sst = sst - np.mean(sst)
    variance = np.std(sst, ddof=1) ** 2
    #print("variance = ", variance)
    # ----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E---------------
    if 0:
        variance = 1.0
        sst = sst / np.std(sst, ddof=1)
    n = len(sst)
    dt = 1/Fs

    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.25  # this will do 4 sub-octaves per octave
    s0 = scale * dt  # this says start at a scale of 10ms, use shorter scale will give you wavelet at high frequecny
    j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.1  # lag-1 autocorrelation for red noise background
    #print("lag1 = ", lag1)
    mother = 'MORLET'
    # Wavelet transform:
    wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
    frequency=1/period
    return sst,frequency,power,global_ws

def plot_wavelet(ax,sst,frequency,power,Fs=10000,colorBar=False,logbase=False):
    import matplotlib.ticker as ticker
    time = np.arange(len(sst)) /Fs   # construct time array
    level=8 #level is how many contour levels you want
    CS = ax.contourf(time, frequency, power, level)
    #ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    #ax.set_title('Wavelet Power Spectrum')
    #ax.set_xlim(xlim[:])
    if logbase:
        ax.set_yscale('log', base=2, subs=None)
    ax.set_ylim([np.min(frequency), np.max(frequency)])
    yax = plt.gca().yaxis
    yax.set_major_formatter(ticker.ScalarFormatter())
    if colorBar: 
        fig = plt.gcf()  # Get the current figure
        position = fig.add_axes([0.2, 0.01, 0.2, 0.01])
        #position = fig.add_axes()
        cbar=plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.05, pad=0.5)
        cbar.set_label('Power (mV$^2$)', fontsize=12) 
        #plt.subplots_adjust(right=0.7, top=0.9)              
    return -1

#%%
dpath='C:/Users/yifan/Downloads/speed_files/Day6/'
filename='Green&Speed_Day6-8.csv'
Fs=840
filepath=os.path.join(dpath,filename)
df = pd.read_csv(filepath)
sample_number=len(df)
midpoint=sample_number//2
#%%
speed_data = df['instant_speed'].values
#%%
#speed_data[speed_data > 500] = 310
# Reshape the 1D array into a 2D array (e.g., 10 rows x 10 columns)
# Make sure that the total number of elements in 'speed_data' can fit into the shape
reshaped_data = speed_data.reshape(1, sample_number)

# Create a heatmap using seaborn
fig, ax = plt.subplots(figsize=(12, 3))

sns.heatmap(reshaped_data, cmap='coolwarm', annot=False, cbar=False,vmin=305, vmax=315)
plt.title("Heatmap of Speed Data")
plt.show()
#%%

fig, ax = plt.subplots(figsize=(12, 3))
zscore_raw = df['raw_z_score']
zscore_bandpass=band_pass_filter(zscore_raw,4,100,Fs)
sst,frequency,power,global_ws=Calculate_wavelet(zscore_bandpass,lowpassCutoff=100,Fs=Fs,scale=20)
plot_wavelet(ax,sst,frequency,power,Fs,colorBar=False,logbase=False)





