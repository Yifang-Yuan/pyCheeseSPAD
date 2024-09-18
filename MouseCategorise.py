# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:49:46 2024

@author: Mingshuai
"""
import os
import pandas as pd
import numpy as np
import re
from scipy import stats
import scipy.signal as signal
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer,StandardScaler

parameter = {
    'grandparent_folder':'E:/Mingshuai/workingfolder/Group E/',
    'pkl_folder':'results',
    'pkl_file_tag':'win_traces',
    'pkl_split_tag': 'Day',
    'pfw_tag': 'PFW_is_',
    'photometry_frame_rate': 130,
    'fft_LB': 0.5,
    'fft_UB': 1,
    'fft_NB': 1
    
    
    
    }

pfw = 0

class col:
    def __init__(self,column):
        self.column = column
        print(column.name)
        num = re.findall(r'\d+', column.name)
        self.trail_ID = int(num[0])
        self.well = int(num[1])
        self.legit = False
        if not self.column.isna().any():
            self.Calculate()
            self.legit = True
    
    def Calculate(self):
        l = len(self.column)
        self.mean_dif = self.column[0:int(l/2)].mean()-self.column[int(l/2):int(l)].mean()
        self.std = np.std(self.column, ddof=1)
        self.fft = ObtainFFT(self.column)
        print(self.fft)
        
class pkl:
    def __init__(self,path,filename):
        self.filename = filename
        self.file = pd.read_pickle(path)
        self.cols1 = []
        self.cols2 = []
        print(filename)
        self.day = int(re.findall(r'\d+', filename.split(parameter['pkl_split_tag'])[1])[0])
        for i in range (self.file.shape[1]):
            column = self.file.iloc[:,i]
            c = col(column)
            if c.well == pfw:
                self.cols1.append(c)
            else:
                self.cols2.append(c)

class mice:
    def __init__(self,folder,mouse_ID):
        self.pkl_files = []
        self.mouse_ID = mouse_ID
        folder = os.path.join(folder,parameter['pkl_folder'])
        for file in os.listdir(folder):
            if 'PFW_is_' in file:
                global pfw
                pfw = int(re.findall(r'\d+', file)[0])
                          
        for file in os.listdir(folder):                 
            if file.endswith('.pkl') and parameter['pkl_file_tag'] in file:
                path = os.path.join(folder,file)
                self.pkl_files.append(pkl(path,file))
        

class group:
    def __init__(self):
        self.mouse = []
        for file in os.listdir(parameter['grandparent_folder']):
            num = re.findall(r'\d+', file)
            if len(num)==0:
                continue
            print('Reading:'+file)
            path = os.path.join(parameter['grandparent_folder'],file)
            self.mouse.append(mice(path,file))

def ObtainFFT (df):
    # Perform FFT
    signal = np.array (df)
    if np.isnan(signal).any():
       signal = signal[~np.isnan(signal)]
     
    fft_values = np.fft.fft(signal)
    fft_frequencies = np.fft.fftfreq(len(signal), 1 / parameter['photometry_frame_rate'])
    
    # Only use the positive half of the spectrum (real signal)
    positive_freq_indices = np.where(fft_frequencies >= 0)
    fft_values = fft_values[positive_freq_indices]
    fft_frequencies = fft_frequencies[positive_freq_indices]
    
    # Extract frequencies in the range of 0.5 Hz to 2 Hz
    freq_range_indices = np.where((fft_frequencies >= parameter['fft_LB']) & (fft_frequencies <=  parameter['fft_UB']))
    fft_values_in_range = fft_values[freq_range_indices]
    fft_frequencies_in_range = fft_frequencies[freq_range_indices]
    #Extract Noise
    freq_noise_indices = np.where(fft_frequencies >= parameter['fft_NB'])
    fft_values_noise = fft_values[freq_noise_indices]
    fft_frequencies_noise = fft_frequencies[freq_noise_indices]
    fft_signal_sum = np.sum(np.abs(fft_values_in_range) ** 2)
    fft_noise_sum = np.sum(np.abs(fft_values_noise) ** 2)
    # fft_tot_sum = np.sum(np.abs(fft_values) ** 2)
    fft_ratio = fft_signal_sum/fft_noise_sum
    return fft_ratio


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster_mice_performance(group, n_clusters=3):
    mouse_features = []
    mouse_ids = []

    # Step 1: Aggregate the performance features for each mouse
    for mice in group.mouse:
        features_list = []
        for pkl_file in mice.pkl_files:
            for col in pkl_file.cols1:
                if col.legit:
                    # Collect features for each trial
                    features_list.append([col.mean_dif, col.fft])
        
        # Step 2: Aggregate the features across all trials for this mouse (e.g., calculate the mean)
        features_array = np.array(features_list)
        mean_features = np.mean(features_array, axis=0)
        std_features = np.std(features_array, axis=0)

        # Combine mean and std into a single feature vector
        aggregated_features = np.hstack([mean_features, std_features])
        mouse_features.append(aggregated_features)
        mouse_ids.append(mice.mouse_ID)

    # Convert to numpy array for clustering
    mouse_features = np.array(mouse_features)

    # Step 3: Standardize the features
    scaler = StandardScaler()
    mouse_features_scaled = scaler.fit_transform(mouse_features)

    # Step 4: Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(mouse_features_scaled)

    # Step 5: Visualize the clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    mouse_features_pca = pca.fit_transform(mouse_features_scaled)
    
    plt.scatter(mouse_features_pca[:, 0], mouse_features_pca[:, 1], c=cluster_labels, cmap='viridis', s=100)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    
    # Annotate points with mouse IDs
    for i, mouse_id in enumerate(mouse_ids):
        plt.text(mouse_features_pca[i, 0], mouse_features_pca[i, 1], mouse_id, fontsize=9, ha='right')
    
    plt.title('KMeans Clustering of Mice Performance')
    plt.xlabel('mean_dif')
    plt.ylabel('fft')
    plt.legend()
    plt.show()

    # Output cluster labels for each mouse
    return dict(zip(mouse_ids, cluster_labels))

# Usage Example:
# Assuming you have a `group` object ready with the necessary data
# cluster_labels = cluster_mice_performance(group, n_clusters=3)
# print("Cluster assignments for each mouse:", cluster_labels)


a = group()
cluster_mice_performance(a)