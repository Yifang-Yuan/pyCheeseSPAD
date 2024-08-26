#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:08:15 2024

@author: zhumingshuai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:03:57 2024

@author: zhumingshuai
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


class input_file:
    def __init__ (self,folder,filename):
        self.folder = folder
        self.filename = filename
        self.ID = filename.split('_')[0]
        self.path = os.path.join(folder,filename)
        self.df = pd.read_csv(self.path)
        self.day_avg = GetDayAvg(self.df)
        if 'Average' in self.filename:
            self.avg = True
        elif 'Route_Score' in self.filename:
            self.avg = False
            if 'Less' in self.filename:
                self.pfw = False
            else:
                self.pfw = True
        else:
            print('ERROR! Please remove irrelevent csv files in '+folder)
        

def ReadFiles (input_folder,SB):
    csv_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.csv'):
                csv_file = input_file(root,filename) 
                if SB and (not('SB_peak_frequency' in csv_file.df.columns)):
                    continue
                csv_files.append(csv_file)
    return csv_files

def IntegrateData (csv_files):
    csv_files_avg = None
    csv_files_norm_pfw = None
    csv_files_norm_lpfw = None
    csv_files_day_avg = None
    for i in range (len(csv_files)):
        file = csv_files[i]
        if (file.avg):
            csv_files_avg = pd.concat([csv_files_avg,file.df], ignore_index=True)
            csv_files_day_avg = pd.concat([csv_files_day_avg,file.day_avg], ignore_index=True)
        else:
            if(file.pfw):
                csv_files_norm_pfw = pd.concat([csv_files_norm_pfw,GetDayAvg(file.df)], ignore_index=True)
            else:
                csv_files_norm_lpfw = pd.concat([csv_files_norm_lpfw,GetDayAvg(file.df)], ignore_index=True)
            
    return csv_files_norm_pfw,csv_files_norm_lpfw,csv_files_avg,csv_files_day_avg

def GetDayAvg (csv_files):
    avg_csv = csv_files.groupby('day').mean().reset_index()
    return avg_csv     
        
def PlotDoubleY (csv_files,y1_column,y2_column,xlab = 'x',ylab1 = 'y1',ylab2 = 'y2',day_column = 'day'):
    mean_df = csv_files.groupby(day_column).mean()
    
    # Calculate the standard error of the mean (SEM)
    std_df = csv_files.groupby(day_column).std()
    count_df = csv_files.groupby(day_column).size()
    count_df = pd.DataFrame({col: count_df for col in std_df.columns})
    sem_df = std_df / np.sqrt(count_df)
    t_value = stats.t.ppf(1-0.025, df=count_df-1)
    ci_df = sem_df * t_value
    days = mean_df.index
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Days')
    ax1.set_ylabel(y1_column, color='black')
    ax1.errorbar(days, mean_df[y1_column], yerr=ci_df[y1_column], fmt='-o', color='black', capsize=5)
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_column, color='green')
    ax2.errorbar(days, mean_df[y2_column], yerr=ci_df[y2_column], fmt='-o', color='green', capsize=5)
    ax2.tick_params(axis='y', labelcolor='green')
    fig.tight_layout()
    return fig

def PlotRSForMultipleMouse(input_folder,output_folder,y1_column,y2_column,PlotSB=False):
    SB = PlotSB
    csv_files = ReadFiles(input_folder,SB)
    for i in range (len(csv_files)):
        if (not csv_files[i].avg) and csv_files[i].pfw:
            op = output_folder+csv_files[i].ID
            if not os.path.exists(op):
                os.makedirs(op)
            fig = PlotDoubleY(csv_files[i].df,y1_column,y2_column)
            fig.savefig(op+'/Preferred_Well_RS.png')
        if (not csv_files[i].avg) and (not csv_files[i].pfw):
            op = output_folder+csv_files[i].ID
            if not os.path.exists(op):
                os.makedirs(op)
            fig = PlotDoubleY(csv_files[i].df,y1_column,y2_column)
            fig.savefig(op+'/Less_Preferred_Well_RS.png')
            
    csv_files_norm_pfw,csv_files_norm_lpfw,csv_files_avg,csv_files_day_avg = IntegrateData(csv_files)
    fig = PlotDoubleY(csv_files_norm_pfw, y1_column,y2_column)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    fig.savefig(output_folder+'Preferred_well_Tot.png')
    
    fig = PlotDoubleY(csv_files_norm_lpfw, y1_column,y2_column)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    fig.savefig(output_folder+'Less_Preferred_well_Tot.png')
    return
                
                
#%%
# input_folder = '/Users/zhumingshuai/Desktop/Programming/Photometry/output/'
# output_folder = '/Users/zhumingshuai/Desktop/Programming/Photometry/output/'
# PlotRSForMultipleMouse(input_folder,output_folder,'route_score', 'z_dif',PlotSB=False)


