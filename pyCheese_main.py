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
session1=pyCheeseSession.pyCheeseSession(pyFolder,COLD_folder,COLD_filename,save_folder,animalID=animalID,SessionID=SessionID,pySBFolder=pySBFolder)
#%%
peak_values,average_peak_value,num_peaks, max_signal_value=session1.find_peaks_in_SBtrials()
#%%
before_window=5
after_window=5
event_window_traces=session1.Plot_multiple_PETH_different_window(before_window,after_window)
#%%
'To batch process multiple days and save event_window_traces'
total_days=5
parent_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/'
COLD_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/workingfolder/'
animalID='1756072'
before_window=5
after_window=5
def ReadMultiDaysSaveEventWindow (total_days,parent_folder,COLD_folder,animalID,before_window,after_window):
    for day in range (1, total_days+1):
        day_py_folder = f'1756072_day{day}/{animalID}CB/'
        COLD_filename=f'Training Data_Day{day}.xlsx'
        SessionID=f'Day{day}'
        full_path = os.path.join(parent_folder, day_py_folder)
        current_session=pyCheeseSession.pyCheeseSession(full_path,COLD_folder,
                                                 COLD_filename,save_folder,animalID=animalID,SessionID=SessionID)
        event_window_traces=session1.Plot_multiple_PETH_different_window(before_window,after_window)
        
        return -1