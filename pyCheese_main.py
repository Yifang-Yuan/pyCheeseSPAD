# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:56:27 2024

@author: Yifang
"""
import pyCheeseSession
import os
#%%
pyFolder='/Volumes/YifangExp/CB_EC5aFibre_1756072/1756072_day1/1756072CB/'
pySBFolder='/Volumes/YifangExp/CB_EC5aFibre_1756072/1756072_day1/1756072SB/'
COLD_folder='/Volumes/YifangExp/CB_EC5aFibre_1756072/COLD_folder/'
# Set your parameters
COLD_filename='Training Data_Day1.xlsx'
save_folder='/Volumes/YifangExp/CB_EC5aFibre_1756072/result'
animalID='1756072'
SessionID='Day1'
session1=pyCheeseSession.pyCheeseSession(pyFolder,COLD_folder,COLD_filename,
                                         save_folder,animalID=animalID,SessionID=SessionID,pySBFolder=pySBFolder)
#%%
'This will plot the average of all collecting reward PETH'
before_window=5
after_window=5
event_window_traces=session1.Plot_multiple_PETH_different_window(before_window,after_window)
#%%
'photometry dataFrame and cheeseboard behaviour result dataFrame of current session'
photometry_df=session1.photometry_df
cheese_df=session1.cheese_df
#%%
'''choose the trial index you want to plot, this is the trial num that you have photometry recordings.
You can check Well1PETH variable for the index'''
for i in range(10):
    singleTrial_index=i
    before_well1_window=5
    after_well2_window=5
    session1.plot_single_trial_2_rewards_PETH(singleTrial_index,before_well1_window, after_well2_window, color='blue', ax=None)
    
