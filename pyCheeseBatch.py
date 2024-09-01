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
import pandas as pd
import Reward_Latency
import re
import MultipleRouteScore
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
        if SB:
            current_session.find_peaks_in_SBtrials(plot=False)
    return -1

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

def plot_2wells_PETH_all_trials (result_folder):
    PSTH_collection=Concat_PETH_pkl_files (result_folder, target_string='win_traces')
    filtered_columns = [col for col in PSTH_collection.columns if col.endswith('_1')]
    Well1_PETH = PSTH_collection[filtered_columns]

    filtered_columns = [col for col in PSTH_collection.columns if col.endswith('_2')]
    Well2_PETH = PSTH_collection[filtered_columns]
    '''plot'''
    fig, ax = plt.subplots(figsize=(10, 4))
    fp.Plot_mean_With_CI_PSTH(Well1_PETH, before_window, after_window, animalID, meancolor='green', stdcolor='lightgreen', ax=ax)

    fig, ax = plt.subplots(figsize=(10, 4))
    fp.Plot_mean_With_CI_PSTH(Well2_PETH, before_window, after_window, animalID, meancolor='red', stdcolor='lightcoral', ax=ax)

    fig, ax = plt.subplots(figsize=(10, 4))
    fp.Plot_mean_With_CI_PSTH(PSTH_collection, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=ax)
    return Well1_PETH,Well2_PETH

def plot_day_average_PETH_together(result_folder):
    PSTH_collection=Concat_PETH_pkl_files (result_folder, target_string='win_traces')
    fig, ax = plt.subplots(figsize=(6, 4))
    filtered_columns = list(filter(lambda col: col.startswith('Day1') and col.endswith('_1'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='red', stdcolor='lightcoral', ax=ax)
    filtered_columns = list(filter(lambda col: col.startswith('Day2') and col.endswith('_1'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='purple', stdcolor='pink', ax=ax)
    filtered_columns = list(filter(lambda col: col.startswith('Day3') and col.endswith('_1'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='green', stdcolor='lightgreen', ax=ax)
    filtered_columns = list(filter(lambda col: col.startswith('Day4') and col.endswith('_1'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=ax)
    filtered_columns = list(filter(lambda col: col.startswith('Day5') and col.endswith('_1'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='black', stdcolor='grey', ax=ax)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    filtered_columns = list(filter(lambda col: col.startswith('Day1') and col.endswith('_2'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='red', stdcolor='lightcoral', ax=ax)
    filtered_columns = list(filter(lambda col: col.startswith('Day2') and col.endswith('_2'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='purple', stdcolor='pink', ax=ax)
    filtered_columns = list(filter(lambda col: col.startswith('Day3') and col.endswith('_2'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='green', stdcolor='lightgreen', ax=ax)
    filtered_columns = list(filter(lambda col: col.startswith('Day4') and col.endswith('_2'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=ax)
    filtered_columns = list(filter(lambda col: col.startswith('Day5') and col.endswith('_2'), PSTH_collection.columns))
    Day_Well1_PETH = PSTH_collection[filtered_columns]
    fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='black', stdcolor='grey', ax=ax)

    return -1
#%%
'This is to call the above function to read all sessions in multiple days for an animal'
grandparent_folder = 'E:/Mingshuai/workingfolder/Group A/Group A (non_cue)/'
output_folder = grandparent_folder+'output/'
parent_list = ['1756072','1746062','1756074']
before_window=5
after_window=5
PlotSB = True
for i in range (len (parent_list)):
    parent_folder = grandparent_folder+parent_list[i]+'/'
    COLD_folder = parent_folder+'Cold_folder/'
    save_folder = parent_folder
    result_folder = parent_folder+'results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    #Obtain total days
    total_days = -1
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
    Read_MultiDays_Save_CB_SB_results (total_days,parent_folder,save_folder,COLD_folder,animalID,before_window,after_window,SB=SB)

    '''plot well1 and well2 average PETH for all sessions'''
    ''' you need to put all the PETH files with the same half window in the same folder '''
    Well1_PETH,Well2_PETH=plot_2wells_PETH_all_trials (result_folder)


    plot_day_average_PETH_together(result_folder)
    Reward_Latency.PlotRouteScoreGraph(COLD_folder,result_folder,output_folder)

MultipleRouteScore.PlotRSForMultipleMouse(output_folder,output_folder,'route_score', 'z_dif')
