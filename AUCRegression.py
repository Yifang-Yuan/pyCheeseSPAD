# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:57:40 2024

@author: mingshuai
"""

import os
import pandas as pd
import numpy as np
import re
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt

parameter = {
    'grandparent_folder':'D:/Photometry/workingfolder/Group E/',
    'pkl_folder':'results',
    'pkl_file_tag':'win_traces',
    'pkl_split_tag': 'Day',
    'cold_folder_tag': 'Cold_folder',
    'cold_split_tag': 'Day',
    'pfw_tag': 'PFW_is_',
    'photometry_frame_rate': 130,
    'fft_LB': 0.5,
    'fft_UB': 1,
    'fft_NB': 1
    }

pfw = 0
output_path = None
mouse_name = None
class col:
    def __init__(self,column):
        self.column = column
        num = re.findall(r'\d+', column.name)
        self.trail_ID = int(num[0])
        self.well = int(num[1])
        self.legit = False
        if not self.column.isna().any():
            self.ObtainAUC()
            self.legit = True
    
    def ObtainAUC(self):
        l = len(self.column)
        seg_a = self.column[0:int(l/2)]
        seg_b = self.column[int(l/2):int(l)]
        self.area1 = np.trapz(seg_a)
        self.area2 = np.trapz(seg_b)
        
class pkl_file:
    def __init__(self,path,filename):
        self.filename = filename
        self.file = pd.read_pickle(path)
        self.cols1 = []
        self.cols2 = []
        self.cols = []
        print(filename)
        self.day = int(re.findall(r'\d+', filename.split(parameter['pkl_split_tag'])[1])[0])
        for i in range (self.file.shape[1]):
            column = self.file.iloc[:,i]
            c = col(column)
            self.cols.append(c)
            if c.well == pfw:
                self.cols1.append(c)
            else:
                self.cols2.append(c)

class single_trail_cold:
    def __init__ (self,row,trail_ID):
        self.row = row
        self.ID = trail_ID
        self.RS1 = row['well1routescore']
        self.RS2 = row['well2routescore']
        if row['firstwellreached'] == 1:
            self.L1 = row['well1time_s']
            self.L2 = row['latencybetweenwells_s']
        else:
            self.L2 = row['well2time_s']
            self.L1 = row['latencybetweenwells_s']
        if pfw == 2:
            self.RS1,self.RS2 = self.RS2,self.RS1
            self.L1,self.L2 = self.L2,self.L1
        
            
class cold_file:
    def __init__(self,file_path,filename):
        print(filename)
        self.cold_file = pd.read_excel(file_path)
        self.day = int(re.findall(r'\d+', filename.split(parameter['cold_split_tag'])[1])[0])
        self.trails_cold = []
        for i in range(self.cold_file.shape[0]):
            row = single_trail_cold(self.cold_file.iloc[i],i)
            self.trails_cold.append(row)

class single_day:
    def __init__(self,pkl,cold):
        self.pkl_file = pkl
        self.cold_file = cold
        self.day = self.pkl_file.day
        self.CalculateMeans()
        
    def CalculateMeans (self):
        self.A1 = []
        self.A2 = []
        self.L = []
        self.RS = []
        for col in self.pkl_file.cols:
            if col.legit:
                self.A1.append(col.area1)
                self.A2.append(col.area2)
                cold = self.cold_file.trails_cold[col.trail_ID]
                if col.well == pfw:
                    self.L.append(cold.L1)
                    self.RS.append(cold.RS1)
                else:
                    self.L.append(cold.L2)
                    self.RS.append(cold.RS2)
        self.A1_mean = np.mean(self.A1)
        self.A2_mean = np.mean(self.A2)
        self.L_mean = np.mean(self.L)
        self.RS_mean = np.mean(self.RS)
        
        
        
class mice:
    def __init__(self,folder,mouse_ID):
        self.pkl_files = []
        self.cold_files = []
        self.mouse_ID = mouse_ID
        self.days = []
        global mouse_name
        mouse_name = self.mouse_ID
        #finding cold folder
        for file in os.listdir(folder):
            if parameter['cold_folder_tag'] in file:
                print(file)
                cold_folder = os.path.join(folder,file)
                break
        
        for file in os.listdir(cold_folder):
            if parameter['cold_split_tag'] in file:
                path = os.path.join(cold_folder,file)
                cold = cold_file(path,file)
                self.cold_files.append(cold)
                
        pkl_folder = os.path.join(folder,parameter['pkl_folder'])
        for file in os.listdir(pkl_folder):
            if 'PFW_is_' in file:
                global pfw
                pfw = int(re.findall(r'\d+', file)[0])
                          
        for file in os.listdir(pkl_folder):                 
            if file.endswith('.pkl') and parameter['pkl_file_tag'] in file:
                path = os.path.join(pkl_folder,file)
                self.pkl_files.append(pkl_file(path,file))
        
        for pkl in self.pkl_files:
            for cold in self.cold_files:
                if (pkl.day == cold.day):
                    day = single_day(pkl,cold)
                    self.days.append(day)
        self.LatencyLinearRegression()
        self.RSLinearRegression()
        
    def LatencyLinearRegression(self):
        path = os.path.join(output_path,'AUC Linear Regression')
        if not os.path.exists(path):
            os.makedirs(path)
        fig1, ax = plt.subplots()
        fig2, bx = plt.subplots()
        A1_tot = []
        A2_tot = []
        L_tot = []
        
        for day in self.days:
            ax.scatter(day.A1, day.L, label='Day'+str(day.day))
            bx.scatter(day.A2, day.L, label='Day'+str(day.day))
            L_tot = np.concatenate([L_tot,day.L])
            A1_tot = np.concatenate([A1_tot,day.A1])
            A2_tot = np.concatenate([A2_tot,day.A2])
        
        ax.legend(loc='upper right')
        ax.set_xlabel('Area Under Curve before reward collection')
        ax.set_ylabel('Latency')
        ax.set_title(mouse_name+' Latency Linear Regression')
        ax = PlotLinearRegression(A1_tot, L_tot, ax)
        fig1.savefig(os.path.join(path, 'Before_Reward_Latency_Regression.png'))
        bx.legend(loc='upper right')
        bx.set_xlabel('Area Under Curve after reward collection')
        bx.set_ylabel('Latency')
        bx.set_title(mouse_name+' Latency Linear Regression')
        bx = PlotLinearRegression(A2_tot, L_tot, bx)
        fig2.savefig(os.path.join(path, 'After_Reward_Latency_Regression.png'))
        
    def RSLinearRegression(self):
        path = os.path.join(output_path,'AUC Linear Regression')
        if not os.path.exists(path):
            os.makedirs(path)
        fig1, ax = plt.subplots()
        fig2, bx = plt.subplots()
        A1_tot = []
        A2_tot = []
        RS_tot = []
        
        for day in self.days:
            ax.scatter(day.A1, day.RS, label='Day'+str(day.day))
            bx.scatter(day.A2, day.RS, label='Day'+str(day.day))
            RS_tot = np.concatenate([RS_tot,day.RS])
            A1_tot = np.concatenate([A1_tot,day.A1])
            A2_tot = np.concatenate([A2_tot,day.A2])
        
        ax.legend(loc='upper right')
        ax.set_xlabel('Area Under Curve before reward collection')
        ax.set_ylabel('Route Score')
        ax.set_title(mouse_name+' RS Linear Regression')
        ax = PlotLinearRegression(A1_tot, RS_tot, ax)
        fig1.savefig(os.path.join(path, 'Before_Reward_RS_Regression.png'))
        bx.legend(loc='upper right')
        bx.set_xlabel('Area Under Curve after reward collection')
        bx.set_ylabel('Route Score')
        bx.set_title(mouse_name+' RS Linear Regression')
        bx = PlotLinearRegression(A2_tot, RS_tot, bx)
        fig2.savefig(os.path.join(path, 'After_Reward_RS_Regression.png'))


class group:
    def __init__(self):
        self.mouse = []
        for file in os.listdir(parameter['grandparent_folder']):
            num = re.findall(r'\d+', file)
            if len(num)==0:
                continue
            print('Reading:'+file)
            path = os.path.join(parameter['grandparent_folder'],file)
            global output_path
            output_path  = path
            self.mouse.append(mice(path,file))
            #Obtain day max for mic
            self.day_max = 9999
        for i in self.mouse:
            day_num = len(i.days)
            if day_num<self.day_max:
                self.day_max = day_num
        print(self.day_max)
    # def LatencyRSLinearRegression(self):
    #     A1 = []
    #     A2 = []
    #     L = []
    #     RS = []
    #     for mice in self.mouse:
    #         for day in mice.days:
                
        
        
def PlotLinearRegression (x,y,ax):
    x = np.array(x)
    y = np.array(y)
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # Generate regression line
    regression_line = slope * x + intercept
    # Plotting the regression line
    ax.plot(x, regression_line, label='Regression line')



a = group()