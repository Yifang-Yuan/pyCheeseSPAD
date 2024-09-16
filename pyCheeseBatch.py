# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:38:13 2024
@author: Yifang
"""
'To batch process multiple days and save event_window_traces'
import matplotlib.pyplot as plt
import pyCheeseSession
import os
import photometry_functions as fp
import FullTracePlot as ftp
import pandas as pd
import Reward_Latency
import re
import MultipleRouteScore
import plotCheese
import heatmap
import numpy as np
from scipy import stats
import scipy.signal as signal

total_days = -1

def Read_MultiDays_Save_CB_SB_results (total_days,parent_folder,save_folder,COLD_folder,animalID,before_window,after_window,SB=True):
    for day in range (1, total_days+1):
        day_py_folder = f'Day{day}_photometry_CB/'
        bonsai_folder = f'Day{day}_Bonsai/'
        if SB==True:
            pySB_string= f'Day{day}_photometry_SB/'
            pySBFolder = os.path.join(parent_folder, pySB_string)
        else:
            pySBFolder=None
        COLD_filename=f'Training_Data_Day{day}.xlsx'
        SessionID=f'Day{day}'
        full_path_CB = os.path.join(parent_folder, day_py_folder)
        bonsai_folder = os.path.join(parent_folder, bonsai_folder)
        current_session=pyCheeseSession.pyCheeseSession(full_path_CB,bonsai_folder,COLD_folder,
                                                 COLD_filename,save_folder,animalID=animalID,SessionID=SessionID,pySBFolder=pySBFolder)
        current_session.Plot_multiple_PETH_different_window(before_window,after_window)
        current_session.Event_time_single_side(window=4)
        current_session.Event_time_two_sides(window=4)
        current_session.StartBox_twosides(before_window=3,after_window=10)
        if SB:
            current_session.find_peaks_in_SBtrials(plot=False)
    return -1

def Integration (mouses,parameter_df,output_folder):
    target_len = (parameter_df['before_win']+parameter_df['after_win'])*parameter_df['frame_rate']
    time = np.arange(0,target_len)/parameter_df['frame_rate']-parameter_df['before_win']
    for Day in range (1,total_days+1):
        plt.clf()
        plt.figure(figsize=(10, 6))
        day_column = 'Day'+str(Day)
        for mouse in mouses:
            PETH_df = pd.DataFrame(columns=['mean', 'UB', 'LB'])
            for i in range (mouse.MouseDf[day_column].shape[0]):
                df = mouse.MouseDf[day_column].iloc[i,:]
                PETH_df = pd.concat([PETH_df,ObtainCI(df)],ignore_index=True)
            plt.plot(time,PETH_df['mean'],label=mouse.ID)
            plt.fill_between(time, PETH_df['LB'], PETH_df['UB'], alpha=0.2, label='95% CI')
            legend = plt.legend(loc='upper right')
            legend.get_frame().set_alpha(0.5)
            plt.xlabel('time/s')
            plt.ylabel('z-score')
            plt.title(day_column+' average PETH')
        plt.savefig(os.path.join(output_folder,day_column+' average PETH.png'))
        plt.close()

# def LowPassFilter (x):
#     # Sampling frequency (in Hz)
#     fs = 130
#     cutoff = 5
#     # Normalized cutoff frequency (cutoff frequency / Nyquist frequency)
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     # Order of the filter
#     order = 5
#     # Get the filter coefficients
#     b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
#     y=signal.filtfilt(b, a, x, axis=0)
#     return y

def PlotFullTrace (parameter_df,parent_folder,COLD_folder,mouse_ID):
    pkl_folder = os.path.join(parent_folder,parameter_df['pkl_folder_tag'])
    for filename in os.listdir(pkl_folder):
        if (filename.endswith('.pkl') and 'full' in filename):
            path = os.path.join(pkl_folder,filename)
            day = int(re.findall(r'\d+', filename.split('Day')[1])[0])
            df = pd.read_pickle(path)
            for i in df.columns:
                trail_ID = int(re.findall(r'\d+', i.split('pyData')[1])[0])
                ftp.Main(df[i],parameter_df,mouse_ID,day,trail_ID,parent_folder,COLD_folder)
            
         
def ObtainCI (df):
    mean = df.mean()
    std = np.std(df)
    n = df.shape[0]
    t_value = stats.t.ppf(0.975, df=n-1)
    CI = t_value*std/np.sqrt(n)
    data = pd.DataFrame({
        'mean':[mean],
        'UB': [mean+CI],
        'LB': [mean-CI]
        })
    return data
#%%
'This is to call the above function to read all sessions in multiple days for an animal'
grandparent_folder = 'E:/Mingshuai/workingfolder/Group E/'
output_folder = grandparent_folder+'output/'
parent_list = ['1084','1086','1105','6534','6535']
before_window=5
after_window=5
PlotSB = True

parameter_df = {
    'frame_rate':130,
    'before_win':5,
    'after_win':5,
    'pkl_folder_tag': 'results',
    'UPthreshold':2,
    'Lowthresold':-2,
    'width':1,
    'sync_parent_tag':'Bonsai',
    'sync_tag':'sync'
    }
mouses = []
for i in range (len (parent_list)):
    parent_folder = grandparent_folder+parent_list[i]+'/'
    
    for filename in os.listdir(parent_folder):
        if 'Cold_folder' in filename or 'COLD_folder' in filename:
            COLD_folder = os.path.join(parent_folder,filename)+'/'
    
    save_folder = parent_folder
    result_folder = parent_folder+'results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    #Obtain total days
    for filename in os.listdir(COLD_folder):
        if 'Day' in filename and filename.endswith('.xlsx'):
            day_str = re.findall(r'\d+', filename.split('Day')[1])
            day = day_str[0]
            day = int(day)
            if(day>total_days):
                total_days = day
    animalID = re.findall(r'\d+', parent_list[i])[-1]
    SB = False
    for dirpath, dirnames, filenames in os.walk(parent_folder):
        if 'SB' in filenames or 'SB' in dirpath:
            SB = True
    if not SB:
        PlotSB = False
    print('Now reading in:'+parent_list[i])
    # Read_MultiDays_Save_CB_SB_results (total_days,parent_folder,save_folder,COLD_folder,animalID,parameter_df['before_win'],parameter_df['after_win'],SB=SB)

    # '''plot well1 and well2 average PETH for all sessions'''
    # ''' you need to put all the PETH files with the same half window in the same folder '''
    # plotCheese.plot_2wells_PETH_all_trials (result_folder,2,2,animalID)
    # plotCheese.plot_day_average_PETH_together(result_folder,2,2,animalID)
    # plotCheese.plot_SB_PETH_all_trials (result_folder,3,5,animalID)
    # plotCheese.plot_day_average_SB_PETH_together (result_folder,3,5,animalID)
    
    Reward_Latency.PlotRouteScoreGraph(COLD_folder,result_folder,output_folder)
    PlotFullTrace(parameter_df,parent_folder,COLD_folder,animalID)
    
    # mouses.append(heatmap.Main(parameter_df,animalID,parent_folder))
    # Integration(mouses,parameter_df,output_folder)
    
    
# MultipleRouteScore.PlotRSForMultipleMouse(output_folder,output_folder,'route_score', 'z_dif')
