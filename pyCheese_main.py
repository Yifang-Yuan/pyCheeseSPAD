# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:56:27 2024

@author: Yifang
"""
import pyCheeseSession
import os
#%%
pyFolder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/1756072_day4/1756072CB/'
pySBFolder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/1756072_day4/1756072SB/'
COLD_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/workingfolder/'
# Set your parameters
COLD_filename='Training Data_Day4.xlsx'
save_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/'
animalID='1756072'
SessionID='Day4'
session1=pyCheeseSession.pyCheeseSession(pyFolder,COLD_folder,COLD_filename,
                                         save_folder,animalID=animalID,SessionID=SessionID,pySBFolder=pySBFolder)
#%%
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
singleTrial_index=7
timediff=df_py_cheese['well2time'+str(singleTrial_index)][0]-df_py_cheese['well1time'+str(singleTrial_index)][0]

fig, ax = plt.subplots(figsize=(10, 4))
fp.Plot_single_trial_PSTH(Well1_PETH, singleTrial_index,timediff,before_window, after_window, animalID, meancolor='b', ax=ax)