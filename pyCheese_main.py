# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:56:27 2024

@author: Yifang
"""
import pyCheeseSession
pyFolder='G:/CB_EC5aFibre/CB_EC5aFibre_1756072/1756072_day4/1756072CB/'
COLD_folder='G:/CB_EC5aFibre/CB_EC5aFibre_1756072/workingfolder/'
# Set your parameters
COLD_filename='Training Data_Day4.xlsx'


recording1=pyCheeseSession.pyCheeseSession(pyFolder,COLD_folder,COLD_filename,animalID='mice1',SessionID='Day1')
photometry_df=recording1.form_pyCheesed_data_to_pandas()