# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:56:27 2024

@author: Yifang
"""
import pyCheeseSession
#%%
pyFolder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/1756072_day4/1756072CB/'
COLD_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/workingfolder/'
# Set your parameters
COLD_filename='Training Data_Day4.xlsx'
save_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/'
session1=pyCheeseSession.pyCheeseSession(pyFolder,COLD_folder,COLD_filename,save_folder,animalID='1756072',SessionID='Day4')
#%%
before_window=5
after_window=5
event_window_traces=session1.Plot_multiple_PETH_different_window(before_window,after_window)
#%%
