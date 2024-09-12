# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:59:35 2024

@author: Mingshuai Zhu
"""

import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import stats
import seaborn

test = None
test2 = None
DD = None
TT = None
class Mouse:
    def __init__(self,ID,parent_folder,parameter_df):
        self.ID = ID
        self.parent_folder = parent_folder
        self.pkl_folder = parameter_df['pkl_folder_tag']
        self.p = parameter_df
        self.pkl_path = os.path.join(self.parent_folder,self.pkl_folder)
        self.ReadIn()
        self.PlotHM()
        
    def ReadIn (self):
        D = []
        T = []
        pre_df = pd.DataFrame()
        for filename in os.listdir(self.pkl_path):
            if filename.endswith('.pkl') and ('win_traces' in filename):
                day = int(re.findall(r'\d+', filename.split('Day')[1])[0])
                pkl_file = pd.read_pickle(os.path.join(self.pkl_path,filename))
                for col in pkl_file.columns:
                    trail_ID = int(re.findall(r'\d+', col.split('pyData')[1])[0])
                    if pkl_file[col].notna().all():
                        pre_df = pd.concat([pre_df, pkl_file[col]], axis=1,ignore_index = True)
                        D.append('Day'+str(day))
                        T.append('Trail'+str(trail_ID))
                        
        header = pd.MultiIndex.from_arrays([D,T])
        self.MouseDf = pd.DataFrame(pre_df.values,columns=header)
        self.MouseDf.columns.names = ['Day', 'Trail']
        global test,test2,DD,TT
        test = pre_df
        test2 = self.MouseDf
        DD = D 
        TT = T

    def PlotHM (self):
        self.target_len = (self.p['before_win']+self.p['after_win'])*self.p['frame_rate']
        self.time = np.arange(0,self.target_len)/self.p['frame_rate']-self.p['before_win']
        plt.figure(figsize=(10, 6))
        upper_bound = int(self.MouseDf.quantile(0.95).max())+1
        lower_bound = int(self.MouseDf.quantile(0.05).min())-1
        seaborn.heatmap(self.MouseDf.T,vmax=3,vmin=-3)
        
        plt.axvline(x=self.target_len/2,color='blue', linestyle='--')
        label = np.arange(-self.p['before_win'],self.p['after_win']+1,step=1)
        tick_positions = np.linspace(0, self.MouseDf.shape[0]-1,len(label))
        
        plt.xticks(ticks = tick_positions,labels=label)
        output_path = self.pkl_path
        output_path = os.path.join(output_path,'Mouse'+str(self.ID)+'_heatmap.png')
        plt.savefig(output_path)
        plt.close()
        return 
    
    
    
def Main (parameter_df,animal_ID,folder):
    return Mouse(animal_ID,folder,parameter_df)

