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

def Read_MultiDays_Save_CB_SB_results (total_days,parent_folder,save_folder,COLD_folder,animalID,before_window,after_window,SB=True):
    for day in range (1, total_days+1):
        day_py_folder = f'{animalID}_day{day}/{animalID}CB/'
        if SB==True:
            pySB_string=f'{animalID}_day{day}/{animalID}SB/'
            pySBFolder = os.path.join(parent_folder, pySB_string)
        else:
            pySBFolder=None
        COLD_filename=f'Training Data_Day{day}.xlsx'
        SessionID=f'Day{day}'
        full_path_CB = os.path.join(parent_folder, day_py_folder)
        current_session=pyCheeseSession.pyCheeseSession(full_path_CB,COLD_folder,
                                                 COLD_filename,save_folder,animalID=animalID,SessionID=SessionID,pySBFolder=pySBFolder)
        current_session.Plot_multiple_PETH_different_window(before_window,after_window)
        if SB:
            current_session.find_peaks_in_SBtrials() 
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
    PSTH_collection=Concat_PETH_pkl_files (result_folder, target_string='traces')
    filtered_columns = [col for col in PSTH_collection.columns if col.endswith('_1')]
    Well1_PETH = PSTH_collection[filtered_columns]

    filtered_columns = [col for col in PSTH_collection.columns if col.endswith('_2')]
    Well2_PETH = PSTH_collection[filtered_columns]
    '''plot'''
    fig, ax = plt.subplots(figsize=(10, 4))
    fp.Plot_mean_With_Std_PSTH(Well1_PETH, before_window, after_window, animalID, meancolor='green', stdcolor='lightgreen', ax=ax)
    fp.Plot_mean_With_CI_PSTH(Well1_PETH, before_window, after_window, animalID, meancolor='green', stdcolor='lightgreen', ax=ax)

    fig, ax = plt.subplots(figsize=(10, 4))
    fp.Plot_mean_With_Std_PSTH(Well2_PETH, before_window, after_window, animalID, meancolor='red', stdcolor='lightcoral', ax=ax)
    fp.Plot_mean_With_CI_PSTH(Well2_PETH, before_window, after_window, animalID, meancolor='red', stdcolor='lightcoral', ax=ax)

    fig, ax = plt.subplots(figsize=(10, 4))
    fp.Plot_mean_With_Std_PSTH(PSTH_collection, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=ax)
    fp.Plot_mean_With_CI_PSTH(PSTH_collection, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=ax)
    return Well1_PETH,Well2_PETH

def plot_day_average_PETH_together(result_folder):

    
    return -1
#%%
'This is to call the above function to read all sessions in multiple days for an animal'
total_days=5
<<<<<<< HEAD
parent_folder='E:/CB_EC5aFibre/CB_EC5aFibre_1746062/'
save_folder='E:/CB_EC5aFibre/CB_EC5aFibre_1746062/'
COLD_folder='E:/CB_EC5aFibre/CB_EC5aFibre_1746062/workingfolder/'
animalID='1746062'
=======
parent_folder='F:/CB_EC5aFibre_1756072/'
save_folder='F:/CB_EC5aFibre_1756072/'
COLD_folder='F:/CB_EC5aFibre_1756072/COLD_folder/'
animalID='1756072'

>>>>>>> 4d7105db09b3d753863b1964bd17d7c43ab999c1
before_window=5
after_window=5
Read_MultiDays_Save_CB_SB_results (total_days,parent_folder,save_folder,COLD_folder,animalID,before_window,after_window,SB=False)
#%%
'''plot well1 and well2 average PETH for all sessions'''
''' you need to put all the PETH files with the same half window in the same folder '''
<<<<<<< HEAD
result_folder='E:/CB_EC5aFibre/CB_EC5aFibre_1756072/results/'
Well1_PETH,Well2_PETH=plot_2wells_PETH_all_trials (result_folder)
#%%
=======
result_folder='F:/CB_EC5aFibre_1756072/results/'
Well1_PETH,Well2_PETH=plot_2wells_PETH_all_trials (result_folder)
#%%
PSTH_collection=Concat_PETH_pkl_files (result_folder, target_string='traces')
fig, ax = plt.subplots(figsize=(6, 4))

# filtered_columns = list(filter(lambda col: col.startswith('Day1') and col.endswith('_1'), PSTH_collection.columns))
# Day_Well1_PETH = PSTH_collection[filtered_columns]
# fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='red', stdcolor='lightcoral', ax=ax)
# filtered_columns = list(filter(lambda col: col.startswith('Day2') and col.endswith('_1'), PSTH_collection.columns))
# Day_Well1_PETH = PSTH_collection[filtered_columns]
# fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=ax)
filtered_columns = list(filter(lambda col: col.startswith('Day3') and col.endswith('_1'), PSTH_collection.columns))
Day_Well1_PETH = PSTH_collection[filtered_columns]
fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='green', stdcolor='lightgreen', ax=ax)
filtered_columns = list(filter(lambda col: col.startswith('Day4') and col.endswith('_1'), PSTH_collection.columns))
Day_Well1_PETH = PSTH_collection[filtered_columns]
fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='purple', stdcolor='pink', ax=ax)
# filtered_columns = list(filter(lambda col: col.startswith('Day5') and col.endswith('_1'), PSTH_collection.columns))
# Day_Well1_PETH = PSTH_collection[filtered_columns]
# fp.Plot_mean_With_CI_PSTH(Day_Well1_PETH, before_window, after_window, animalID, meancolor='black', stdcolor='grey', ax=ax)
>>>>>>> 4d7105db09b3d753863b1964bd17d7c43ab999c1
