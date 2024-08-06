# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 20:22:58 2023
@author: Yifang

This code is to analyse pyPhotometry recorded data when a mouse is performing cheeseboard task for a multiple trials.
A sync pulse was send to pyPhotometry Digital 1 as CamSync
An LED pulse was recorded by the camera to sync the photometry signal with the animal's behaviour.

Analysing of behaviour was performed with COLD-a cheeseboard behaviour tracking pipeline developed by Daniel-Lewis Fallows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import photometry_functions as fp
import os
# Folder with your files
#folder = 'C:/SPAD/pyPhotometry_v0.3.1/data/' # Modify it depending on where your file is located
#folder ='E:/Qingren/1002/Day2/'
#COLD_folder='E:/Qingren/'
folder ='G:/CB_EC5aFibre/CB_EC5aFibre_1756072/1756072_day4/1756072CB/'
COLD_folder='G:/CB_EC5aFibre/CB_EC5aFibre_1756072/workingfolder/'
# Set your parameters
COLD_filename='Training Data_Day4.xlsx'
pyFs=130
CamFs=24
#half_timewindow=1
animalID='(Mouse 1756072 GCcamp8)'
#%%
#cheeaseboard_session_data=fp.read_cheeseboard_from_COLD (COLD_folder, COLD_filename)
'''This function will read all photometry recording and COLD file for a session or multiple trials'''
df_py_cheese=fp.read_all_photometry_files(folder, '2024-','sync',CamFs,pyFs,COLD_folder,COLD_filename)
#%%
#Enter the value 
before_window=5
after_window=5
event_window_traces=fp.Plot_multiple_PETH_different_window(df_py_cheese,before_window,after_window,fs=pyFs,animalID=animalID)
'''save the pkl file for the PETH data with half window time specified'''
filename='1665D4PETH_'+str(before_window)+'seconds_day4.pkl'
event_window_traces.to_pickle(os.path.join(folder, filename))
#%%