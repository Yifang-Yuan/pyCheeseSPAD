# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:06:22 2024

@author: s2764793
"""
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

def Findpeaks(array,parameter_df):
    w = parameter_df['width']*parameter_df['frame_rate']
    Uppeak,x = find_peaks(array,height=parameter_df['UPthreshold'],width = w)
    Lowpeak,x = find_peaks(-array,height=parameter_df['UPthreshold'],width = w)
    return Uppeak,Lowpeak

def Sync(parent_folder,day,parameter_df,trail_ID):
    
    for filename in os.listdir(parent_folder):
        if 'Day'+str(day) in filename and parameter_df['sync_parent_tag'] in filename:
            sync_folder = os.path.join(parent_folder,filename)
            break
    
    for filename in os.listdir(sync_folder):
        if parameter_df['sync_tag'] in filename:
            trail = int(re.findall(r'\d+', filename.split(parameter_df['sync_tag'])[1])[0])
            if trail == trail_ID:
                sync_file = pd.read_csv(os.path.join(sync_folder,filename))
        
    for i in range (sync_file.shape[0]):
        if np.isnan(sync_file['Value.X'][i]) and np.isnan(sync_file['Value.Y'][i]):
            sync_time = i/parameter_df['frame_rate']
            break
    return sync_time
    

def Main (array,parameter_df,mouse_ID,day,trail_ID,parent_folder,cold_folder):
    U,L = Findpeaks(array,parameter_df)
    w = parameter_df['width']*parameter_df['frame_rate']
    target_len = len(array)
    time = np.arange(0,target_len)/parameter_df['frame_rate']
    
    plt.figure(figsize=(30, 10))
    plt.plot(time,array)
    y_min, y_max = plt.ylim()
    for i in U:
        x1 = (i-w/2)/parameter_df['frame_rate']
        x2 = (i+w/2)/parameter_df['frame_rate']
        plt.fill_betweenx([y_min, y_max], x1, x2, color='blue', alpha=0.5)
    
    for i in L:
        x1 = (i-w/2)/parameter_df['frame_rate']
        x2 = (i+w/2)/parameter_df['frame_rate']
        
        plt.fill_betweenx([y_min, y_max], x1, x2, color='red', alpha=0.5)
    
    plt.xlabel('time/s')
    plt.ylabel('z-score')
    plt.title(mouse_ID+'_Day'+str(day)+'-'+str(trail_ID))
    
    sync_time = Sync(parent_folder,day,parameter_df,trail_ID)
    for filename in os.listdir(cold_folder):
        if 'Day'+str(day) in filename and filename.endswith('.xlsx'):
            cold = pd.read_excel(os.path.join(cold_folder,filename))
            break
    starting = cold['startingtime_s'][trail_ID]-sync_time
    pfw = cold['firstwellreached'][trail_ID]
    fw = cold['well1time_s'][trail_ID]+starting
    sw = cold['well2time_s'][trail_ID]+starting
    leaving = cold['leftfirstwell_s'][trail_ID]+starting
    
    if starting>0 and starting<120:
        plt.axvline(starting,label='enter',linestyle='--',color='green')
    if fw>0 and fw<120:
        plt.axvline(fw,label='well1',linestyle='--',color='r')
    if sw>0 and sw<120:
        plt.axvline(sw,label='well2',linestyle='--',color='r')
    if leaving>0 and leaving<120:
        plt.axvline(leaving,label='leave',linestyle='--',color='purple')
    
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_alpha(0.5)
    
    output_path = os.path.join(parent_folder,'FullTrace')
    output_path = os.path.join(output_path,'Day'+str(day))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path,mouse_ID+'_Day'+str(day)+'-'+str(trail_ID))+'.png')
    plt.close()
    return 