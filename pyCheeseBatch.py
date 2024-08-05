# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:38:13 2024

@author: Yifang
"""
'To batch process multiple days and save event_window_traces'

import pyCheeseSession
import os
def Read_MultiDays_Save_CB_SB_results (total_days,parent_folder,save_folder,COLD_folder,animalID,before_window,after_window,SB=True):
    for day in range (1, total_days+1):
        day_py_folder = f'1756072_day{day}/{animalID}CB/'
        if SB==True:
            pySB_string=f'1756072_day{day}/{animalID}SB/'
            pySBFolder = os.path.join(parent_folder, pySB_string)
        else:
            pySBFolder=None
        COLD_filename=f'Training Data_Day{day}.xlsx'
        SessionID=f'Day{day}'
        full_path_CB = os.path.join(parent_folder, day_py_folder)
        current_session=pyCheeseSession.pyCheeseSession(full_path_CB,COLD_folder,
                                                 COLD_filename,save_folder,animalID=animalID,SessionID=SessionID,pySBFolder=pySBFolder)
        current_session.Plot_multiple_PETH_different_window(before_window,after_window)    
        current_session.find_peaks_in_SBtrials() 
    return -1
    


total_days=5
parent_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/'
save_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/'
COLD_folder='D:/CB_EC5aFibre/CB_EC5aFibre_1756072/workingfolder/'
animalID='1756072'
before_window=5
after_window=5
Read_MultiDays_Save_CB_SB_results (total_days,parent_folder,save_folder,COLD_folder,animalID,before_window,after_window)
