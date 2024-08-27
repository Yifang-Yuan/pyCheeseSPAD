#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:01:51 2024

@author: zhumingshuai
"""
import pandas as pd
import os
import Reward_Latency as RL
import matplotlib.pyplot as plt
import numpy as np

class MousePair:
    def __init__(self,cue_mouse,non_cue_mouse):
        self.cue_mouse = cue_mouse
        self.non_cue_mouse = non_cue_mouse

    def Plot(self,file_type,column_name,title='',ylab = None):
        if not ylab:
            ylab = column_name
        fig,ax = plt.subplots(figsize=(7, 5))
        if file_type == 'lag':
            df1 = self.cue_mouse.lag_file[~np.isnan(self.cue_mouse.lag_file[column_name])]
            df2 = self.non_cue_mouse.lag_file[~np.isnan(self.non_cue_mouse.lag_file[column_name])]
            RL.PlotLagDif(df1,'day',column_name,ax,color='black', xlab = 'Day',ylab = column_name,label='with cue',axh = True)
            RL.PlotLagDif(df2,'day',column_name,ax,color='green', xlab = 'Day',ylab = column_name,label='without cue',title=title,axh = True)
        elif file_type == 'pf':
            df1 = self.cue_mouse.pf_file
            df2 = self.non_cue_mouse.pf_file
            RL.PlotLagDif(df1,'day',column_name,ax,color='black', xlab = 'Day',ylab = column_name,label='with cue')
            RL.PlotLagDif(df2,'day',column_name,ax,color='green', xlab = 'Day',ylab = column_name,label='without cue',title=title)
        elif file_type == 'lpf':
            df1 = self.cue_mouse.lpf_file
            df2 = self.non_cue_mouse.lpf_file
            RL.PlotLagDif(df1,'day',column_name,ax,color='black', xlab = 'Day',ylab = column_name,label='with cue')
            RL.PlotLagDif(df2,'day',column_name,ax,color='green', xlab = 'Day',ylab = column_name,label='without cue',title=title)
        elif file_type == 'avg':
            df1 = self.cue_mouse.avg_file
            df2 = self.non_cue_mouse.avg_file
            RL.PlotLagDif(df1,'day',column_name,ax,color='black', xlab = 'Day',ylab = column_name,label='with cue')
            RL.PlotLagDif(df2,'day',column_name,ax,color='green', xlab = 'Day',ylab = column_name,label='without cue',title=title)
        
        fig.savefig(self.output_folder+column_name+'.png')
    
    def Comparision (self,output_folder):
        self.output_folder = output_folder
        self.Plot('lag','Lag_dif1',title='Comparision of change in signal after collecting the preferred well',ylab='change of mean z-score before and after collection')
        self.Plot('lag','Lag_dif2',title='Comparision of change in signal after collecting the less preferred well',ylab='change of mean z-score before and after collection')
        self.Plot('pf','route_score',title='Comparision of route score for preferred well')
        self.Plot('lpf','route_score',title='Comparision of route score for less preferred well')
        
class Mouse:
    def __init__ (self,folder,ID):
        self.folder = folder
        self.ID = ID
        self.ContainSB = False
        self.ReadFiles()
        
    def ReadFiles(self):
        for filename in os.listdir(self.folder):
            if not filename.endswith('.csv'):
                continue
            if 'Lag' in filename:
                self.lag_file = pd.read_csv(self.folder+filename)
            elif 'Less_Preferred' in filename:
                self.lpf_file = pd.read_csv(self.folder+filename)
            elif 'Preferred' in filename:
                self.pf_file = pd.read_csv(self.folder+filename)
            elif 'Average' in filename:
                self.avg_file = pd.read_csv(self.folder+filename)
                for i in range (len(self.avg_file.columns.tolist())):
                    if 'SB' in self.avg_file.columns[i]:
                        self.ContainSB = True
        return

def MainFunction(cue_grandpa_folder, non_cue_grandpa_folder,output_folder):
    ID_list = []
    cue_folder = cue_grandpa_folder+'output/'
    non_cue_folder = non_cue_grandpa_folder+'output/'
    for filename in os.listdir(cue_folder):
        if (not '.' in filename) and (not 'learning' in filename):
            ID_list.append(filename)
    
    mouse_pair_list = []
    for i in range (len(ID_list)):
        cue_mouse = Mouse(cue_folder+ID_list[i]+'/',ID_list[i])
        non_cue_mouse = Mouse(non_cue_folder+ID_list[i]+'/',ID_list[i])
        mouse_pair = MousePair(cue_mouse,non_cue_mouse)
        mouse_pair_list.append(mouse_pair)
        mouse_pair.Comparision(output_folder)
    
    
    return

#%%
cue_grandpa_folder = '/Volumes/YifangExp/Mingshuai/workingfolder/Group A/Group A (cue)/'
non_cue_grandpa_folder = '/Volumes/YifangExp/Mingshuai/workingfolder/Group A/Group A (non_cue)/'
output_folder = '/Volumes/YifangExp/Mingshuai/workingfolder/Group A/comparision/'
MainFunction(cue_grandpa_folder, non_cue_grandpa_folder,output_folder)