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
import plotCheese

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
    return current_session


#%%
'This is to call the above function to read all sessions in multiple days for an animal'
grandparent_folder = 'G:/CheeseboardYY/GCaMP8m/'
output_folder = grandparent_folder+'output/'
parent_list = ['1804115']
before_window=5
after_window=5
PlotSB = True
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
    current_session=Read_MultiDays_Save_CB_SB_results (total_days,parent_folder,save_folder,COLD_folder,animalID,before_window,after_window,SB=SB)

    '''plot well1 and well2 average PETH for all sessions'''
    ''' you need to put all the PETH files with the same half window in the same folder '''
#     plotCheese.plot_2wells_PETH_all_trials (result_folder,2,2,animalID)
#     plotCheese.plot_day_average_PETH_together(result_folder,2,2,animalID)
#     plotCheese.plot_SB_PETH_all_trials (result_folder,3,5,animalID)
#     plotCheese.plot_day_average_SB_PETH_together (result_folder,3,5,animalID)
        
#     Reward_Latency.PlotRouteScoreGraph(COLD_folder,result_folder,output_folder)

# MultipleRouteScore.PlotRSForMultipleMouse(output_folder,output_folder,'route_score', 'z_dif')
