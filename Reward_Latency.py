#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:44:04 2024

@author: zhumingshuai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:33:37 2024

@author: zhumingshuai
"""

import matplotlib.pyplot as plt
import photometry_functions as fp
import pandas as pd
from scipy.stats import linregress
import numpy as np
import os
import re

#If the well latency is greater than the critical_ignore_time. It will be mark as not found
critical_ignore_time = 120

class trail_route_score:
    z_min_avg = None
    z_max_avg = None
    route_score_avg = None
    z_min_1 = None
    z_max_1 = None
    z_min_2 = None
    z_max_2 = None
    pct_h1 = None
    pct_l1 = None
    pct_h2 = None
    pct_l2 = None
    z_dif_1 = None
    z_dif_2 = None
    
    def __init__ (self,well1_route_score,well2_route_score,well1_latency,well2_latency,z1,z2,day,SB_avg_PV,SB_NP,SB_zdff_max):
        self.well1_route_score = well1_route_score
        self.well2_route_score = well2_route_score
        self.well1_latency = well1_latency
        self.well2_latency = well2_latency
        self.z1 = z1
        self.z2 = z2
        self.day = day
        self.SB_NP = SB_NP
        self.SB_avg_PV = SB_avg_PV
        self.SB_zdff_max = SB_zdff_max
        
    def Calculate(self,pct):
        self.z_min_1 = 999999
        self.z_max_1 = -1
        self.z_min_2 = 999999
        self.z_max_2 = -1
        for j in range (self.z1.shape[0]):
            element_1 = self.z1.iloc[j]
            element_2 = self.z2.iloc[j]
            if (element_1>self.z_max_1):
                self.z_max_1 = element_1
            if (element_1<self.z_min_1):
                self.z_min_1 = element_1
            if (element_2>self.z_max_2):
                self.z_max_2 = element_2
            if (element_2<self.z_min_2):
                self.z_min_2 = element_2
        self.z_min_avg = (self.z_min_1+self.z_min_2)/2
        self.z_max_avg = (self.z_max_1+self.z_max_2)/2
        self.route_score_avg = (self.well1_route_score+self.well2_route_score)/2
        self.pct_h1 = np.percentile(self.z1,100-pct)
        self.pct_l1 = np.percentile(self.z1,pct)
        self.pct_h2 = np.percentile(self.z2,100-pct)
        self.pct_l2 = np.percentile(self.z2,pct)
        z1_len = int(self.z1.shape[0]/2)
        z2_len = int(self.z2.shape[0]/2)
        z_1_a = self.z1.iloc[:z1_len]
        z_1_b = self.z1.iloc[z1_len:]
        z_1_a_max = np.percentile(z_1_a,100-pct)
        z_1_b_min = np.percentile(z_1_b,pct)
        self.z_dif_1 = z_1_a_max-z_1_b_min
        z_2_a = self.z2.iloc[:z2_len]
        z_2_b = self.z2.iloc[z2_len:]
        z_2_a_max = np.percentile(z_2_a,100-pct)
        z_2_b_min = np.percentile(z_2_b,pct)
        self.z_dif_2 = z_2_a_max-z_2_b_min
        return   

class trails_tot:
   def __init__(self):
       self.route_score = []
       self.z_min = []
       self.z_max = []
       self.z_min_avg = []
       self.z_max_avg = []
       self.route_score_avg = []
       self.pct_high = []
       self.pct_low = []
       self.z_dif = []
       self.SB_NP = []
       self.SB_avg_PV = []
       self.SB_zdff_max = []
       self.SB_NP_U = []
       self.SB_avg_PV_U = []
       self.SB_zdff_max_U = []
       self.trail_ID = []
       self.z_dif_avg = []
       self.day = []
       self.day_avg = []
       self.latency = []
       self.is_preferred_well = []
       self.trail_ID_avg = []

class truncated_data:
    def __init__ (self,z_min,z_max,route_score,z_min_avg,z_max_avg,route_score_avg,pct_high,pct_low,z_dif,SB_avg_PV,SB_NP,SB_zdff_max,SB_avg_PV_U,SB_NP_U,SB_zdff_max_U,trail_ID,trail_ID_avg,z_dif_avg,day_ID,day_ID_avg,latency,is_preferred_well):
        self.z_min = z_min
        self.z_max = z_max
        self.route_score = route_score
        self.z_min_avg = z_min_avg
        self.z_max_avg = z_max_avg
        self.route_score_avg = route_score_avg
        self.pct_high = pct_high
        self.pct_low = pct_low
        self.z_dif = z_dif
        self.SB_NP = SB_NP
        self.SB_avg_PV = SB_avg_PV
        self.SB_zdff_max = SB_zdff_max
        self.SB_NP_U = SB_NP_U
        self.SB_avg_PV_U = SB_avg_PV_U
        self.SB_zdff_max_U = SB_zdff_max_U
        self.trail_ID = trail_ID
        self.trail_ID_avg = trail_ID_avg
        self.z_dif_avg = z_dif_avg
        self.day_ID = day_ID
        self.day_ID_avg = day_ID_avg
        self.latency = latency
        self.is_preferred_well = is_preferred_well
    
def ReadRouteScore (cold_folder, cold_file,pickle_folder,pickle_file,day,pfw,percentile):
    route_score_array = []
    route_score_input = fp.read_cheeseboard_from_COLD(cold_folder, cold_file)
    input_z_score = pd.read_pickle(pickle_folder+pickle_file)
    SB_avg_PV = None
    SB_NP = None
    SB_zdff_max = None

    for i in range (route_score_input.shape[0]):
        SB_filename = 'Day'+str(day)+'_trial'+str(i)
        for filename in os.listdir(pickle_folder):
            if (SB_filename in filename) and ('SB' in filename):
                SB = pd.read_pickle(pickle_folder+filename)
                SB_avg_PV = SB['average_peak_value']
                SB_NP = SB['peak_freq']
                SB_zdff_max = SB['zdff_max']
        well1_route_score = route_score_input['well1routescore'][i]
        well2_route_score = route_score_input['well2routescore'][i]
        filtered_z = input_z_score.filter(like='pyData'+str(i))
        for j in range (filtered_z.shape[1]):
            if (filtered_z.columns[j][-1]=='1'):
                z1 = filtered_z.iloc[:,j]
            if (filtered_z.columns[j][-1]=='2'):
                z2 = filtered_z.iloc[:,j]
        well1_latency = np.nan
        well2_latency = np.nan
        if (route_score_input['firstwellreached'][i]==2):
            z1,z2 = z2,z1
            well2_latency = route_score_input['well2time_s'][i]
            well1_latency = route_score_input['latencybetweenwells_s'][i]
        elif (route_score_input['firstwellreached'][i]==1):
            well1_latency = route_score_input['well1time_s'][i]
            well2_latency = route_score_input['latencybetweenwells_s'][i]
         
        if well1_latency>=critical_ignore_time:
            well1_latency = np.nan
        if well2_latency>=critical_ignore_time:
            well2_latency = np.nan
        
        if pfw == 2:
            z1,z2 = z2,z1
            well1_latency,well2_latency = well2_latency,well1_latency
            well1_route_score,well2_route_score = well2_route_score,well1_route_score
        single_trail_score = trail_route_score(well1_route_score,well2_route_score,well1_latency,well2_latency,z1,z2,day,SB_avg_PV,SB_NP,SB_zdff_max)
        single_trail_score.Calculate(percentile)
        route_score_array.append(single_trail_score)    
    
    
    return route_score_array


def Plot_Single_RS (route_score_array,day,output_folder,SB=True):
    z_min = []
    z_max = []
    route_score = []
    route_score_avg = []
    z_min_avg =[]
    z_max_avg = []
    pct_high = []
    pct_low = []
    z_dif = []
    SB_NP = []
    SB_avg_PV = []
    SB_zdff_max = []
    SB_avg_PV_U = []
    SB_NP_U = []
    SB_zdff_max_U = []
    trail_ID = []
    z_dif_avg = []
    day_ID = []
    day_ID_avg = []
    latency = []
    is_preferred_well = []
    trail_ID_avg = []
    filename_prefix = 'Route_Score_Plot_Day'
    for i in range (len(route_score_array)):
        if ((not np.isnan(route_score_array[i].well1_route_score)) and not(np.isnan(route_score_array[i].z_dif_1)) and not (np.isnan(route_score_array[i].z1[0])) and not(np.isnan(route_score_array[i].well1_latency))):
            z_min.append(route_score_array[i].z_min_1)
            z_max.append(route_score_array[i].z_max_1)
            pct_high.append(route_score_array[i].pct_h1)
            pct_low.append(route_score_array[i].pct_l1)
            z_dif.append(route_score_array[i].z_dif_1)
            day_ID.append(int(day))
            trail_ID.append(i)
            route_score.append(route_score_array[i].well1_route_score)
            latency.append(route_score_array[i].well1_latency)
            is_preferred_well.append(True)
            if (SB):
                SB_NP_U.append(route_score_array[i].SB_NP)
                SB_avg_PV_U.append(route_score_array[i].SB_avg_PV)
                SB_zdff_max_U.append(route_score_array[i].SB_zdff_max)
        if (not np.isnan(route_score_array[i].well2_route_score) and not(np.isnan(route_score_array[i].z_dif_2)) and not (np.isnan(route_score_array[i].z2[0])) and not(np.isnan(route_score_array[i].well2_latency))):
            z_min.append(route_score_array[i].z_min_2)
            z_max.append(route_score_array[i].z_max_2)
            pct_high.append(route_score_array[i].pct_h2)
            pct_low.append(route_score_array[i].pct_l2)
            route_score.append(route_score_array[i].well2_route_score)
            z_dif.append(route_score_array[i].z_dif_2)
            trail_ID.append(i)
            day_ID.append(day)
            latency.append(route_score_array[i].well2_latency)
            is_preferred_well.append(False)
            if (SB):
                SB_NP_U.append(route_score_array[i].SB_NP)
                SB_avg_PV_U.append(route_score_array[i].SB_avg_PV)
                SB_zdff_max_U.append(route_score_array[i].SB_zdff_max)
        if ((not np.isnan(route_score_array[i].well2_route_score)) and (not np.isnan(route_score_array[i].well1_route_score)) and not(np.isnan(route_score_array[i].z_dif_1)) and not (np.isnan(route_score_array[i].z1[0])) and not (np.isnan(route_score_array[i].z2[0])) and not(np.isnan(route_score_array[i].well2_latency)) and not(np.isnan(route_score_array[i].well1_latency))):
            z_max_avg.append(route_score_array[i].z_max_avg)
            z_min_avg.append(route_score_array[i].z_min_avg)
            trail_ID_avg.append(i)
            if (SB):
                SB_NP.append(route_score_array[i].SB_NP)
                SB_avg_PV.append(route_score_array[i].SB_avg_PV)
                SB_zdff_max.append(route_score_array[i].SB_zdff_max)
            route_score_avg.append(route_score_array[i].route_score_avg)
            z_dif_avg.append((route_score_array[i].z_dif_2+route_score_array[i].z_dif_1)/2)
            day_ID_avg.append(int(day))
            
    fig = plt.figure(figsize=(10, 30))
    ax1 = fig.add_subplot(711)
    ax2 = fig.add_subplot(712)
    ax3 = fig.add_subplot(713)
    ax4 = fig.add_subplot(714)
    ax5 = fig.add_subplot(715)
    ax6 = fig.add_subplot(716)
    ax7 = fig.add_subplot(717)
    if (len(np.array(route_score))!=0):
        PlotLinearRegression(ax1, np.array(route_score), z_min, y_label='z_min')
        PlotLinearRegression(ax2, np.array(route_score), z_max, y_label='z_max')
        PlotLinearRegression(ax5, np.array(route_score), pct_high, y_label='z_percentile_high',x_label='route_score')
        PlotLinearRegression(ax6, np.array(route_score), pct_low, y_label='z_percentile_low',x_label='route_score')
        PlotLinearRegression(ax7, np.array(route_score), z_dif, y_label='z_dif',x_label='route_score')
    if (len(np.array(route_score_avg))!=0):
        PlotLinearRegression(ax3, np.array(route_score_avg), z_min_avg, y_label='z_min_average',x_label='route_score_average')
        PlotLinearRegression(ax4, np.array(route_score_avg), z_max_avg, y_label='z_max_average',x_label='route_score_average')
    
    if (len(SB_NP)!=0):
        fig_SB = plt.figure(figsize=(10, 20))
        SB_ax1 = fig_SB.add_subplot(311)
        SB_ax2 = fig_SB.add_subplot(312)
        SB_ax3 = fig_SB.add_subplot(313)
        PlotLinearRegression(SB_ax1, np.array(route_score_avg), SB_NP, y_label='SB_peak_frequency',x_label='route_score_average')
        PlotLinearRegression(SB_ax2, np.array(route_score_avg), SB_avg_PV, y_label='SB_average_peak_value',x_label='route_score_average')
        PlotLinearRegression(SB_ax3, np.array(route_score_avg), SB_zdff_max, y_label='SB_zdff_max',x_label='route_score_average')
    
        fig_SB_time = plt.figure(figsize=(10, 20))
        SB_bx1 = fig_SB_time.add_subplot(311)
        SB_bx2 = fig_SB_time.add_subplot(312)
        SB_bx3 = fig_SB_time.add_subplot(313)
        SB_bx1.plot(trail_ID_avg, SB_NP, marker='o', linestyle='-', color='b')
        SB_bx2.plot(trail_ID_avg, SB_avg_PV, marker='o', linestyle='-', color='b')
        SB_bx3.plot(trail_ID_avg, SB_zdff_max, marker='o', linestyle='-', color='b')
        SB_bx1.set_xlabel('number of trail')
        SB_bx2.set_xlabel('number of trail')
        SB_bx3.set_xlabel('number of trail')
        SB_bx1.set_ylabel('peak frequency in SB')
        SB_bx2.set_ylabel('average_peak_value_in_SB')
        SB_bx3.set_ylabel('zdff_max')
        fig_SB_time.savefig(output_folder+'SB_with_time_'+filename_prefix+str(day))
        fig_SB.savefig(output_folder+'SB_'+filename_prefix+str(day))
        
    fig.savefig(output_folder+'CB_'+filename_prefix+str(day))
    return truncated_data(z_min, z_max, route_score, z_min_avg, z_max_avg, route_score_avg,pct_high,pct_low,z_dif,SB_avg_PV,SB_NP,SB_zdff_max,SB_avg_PV_U,SB_NP_U,SB_zdff_max_U,trail_ID,trail_ID_avg,z_dif_avg,day_ID,day_ID_avg,latency,is_preferred_well)

def PlotLinearRegression (ax , x, y, x_label = 'route score',y_label = 'y',name = 'Linear regression for route score'):
    ax.scatter(x,y)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    regression_line = slope * x + intercept
    ax.plot(x, regression_line, color='red', label='Regression Line')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return

def AddVerticalLines (ax, axvline):
    for i in range (len(axvline)):
        ax.axvline(x=axvline[i], color='r', linestyle='--', linewidth=2)
    return ax

def PlotRouteScoreGraph (cold_folder, pickle_folder, output_folder,percentile=5):
    SB = False
    tot_trails = trails_tot()
    day_max = -1
    #obtain the maximum day
    for filename in os.listdir(pickle_folder):
        if filename.endswith('.pkl') or filename.endswith('.xlsx'):
            day = re.findall(r'\d+', filename.split('Day')[1])
            day = day[0]
            day = int(day)
            if(day>day_max):
                day_max = day
    files = [[None for _ in range(2)] for _ in range(day_max)]
    
    for filename in os.listdir(pickle_folder):
        if filename.endswith('.pkl'):
            day = re.findall(r'\d+', filename.split('Day')[1])
            day = day[0]
            day = int(day)
            if 'win' in filename:
                files[day-1][1] = filename
            if ('SB' in filename):
                SB = True
                
    for filename in os.listdir(cold_folder):
        if (filename.endswith('.xlsx')) and ('Day' in filename):
            day = re.findall(r'\d+', filename.split('Day')[1])
            day = day[0]
            day = int(day)
            files[day-1][0] = filename
    total_route_score_array = []
    
    if (not SB):
        print('Warning! No data from starting box found in the pickle file folder:')
        print(pickle_folder)
    
    w1 = 0
    w2 = 0
    pfw = 0
    #obtain the preferred well 
    for i in range (len(files)):
        route_score_input = fp.read_cheeseboard_from_COLD(cold_folder, files[i][0])
        for j in range (route_score_input.shape[0]):
            if (route_score_input['firstwellreached'][j]==1):
                w1+=1
            elif (route_score_input['firstwellreached'][j]==2):
                w2+=1

    if w1>w2:
        pfw = 1
        print('the preferred well is 1')
    else:
        pfw = 2
        print('the preferred well is 2')
        
    
    for i in range (len(files)):
        route_score_array = ReadRouteScore(cold_folder, files[i][0],pickle_folder,files[i][1],i+1,pfw,percentile=percentile)
        total_route_score_array.append(route_score_array)
    
    mouse_ID = files[0][1].split('_')[0]
    output_folder = output_folder+mouse_ID+'/'
    single_plot_output = output_folder+'Single_Day_Plot/'
    if not os.path.exists(single_plot_output):
        os.makedirs(single_plot_output)
    
 #integrate data
    axvline = []
    neo_day = 0
    for i in range (len(total_route_score_array)):
        td = Plot_Single_RS(total_route_score_array[i], i+1 , single_plot_output,SB=SB)
        tot_trails.z_min.extend(td.z_min)
        tot_trails.z_max.extend(td.z_max)
        tot_trails.route_score.extend(td.route_score)
        tot_trails.z_min_avg.extend(td.z_min_avg)
        tot_trails.z_max_avg.extend(td.z_max_avg)
        tot_trails.route_score_avg.extend(td.route_score_avg)
        tot_trails.pct_high.extend(td.pct_high)
        tot_trails.pct_low.extend(td.pct_low)
        tot_trails.z_dif.extend(td.z_dif)
        tot_trails.SB_NP_U.extend(td.SB_NP_U)
        tot_trails.SB_avg_PV_U.extend(td.SB_avg_PV_U)
        tot_trails.SB_zdff_max_U.extend(td.SB_zdff_max_U)
        if (SB):
            tot_trails.SB_NP.extend(td.SB_NP)
            tot_trails.SB_avg_PV.extend(td.SB_avg_PV)
            tot_trails.SB_zdff_max.extend(td.SB_zdff_max)
        tot_trails.z_dif_avg.extend(td.z_dif_avg)
        tot_trails.day.extend(td.day_ID)
        tot_trails.day_avg.extend(td.day_ID_avg)
        tot_trails.latency.extend(td.latency)
        tot_trails.is_preferred_well.extend(td.is_preferred_well)
        neo_ID = np.array(td.trail_ID)+neo_day
        neo_ID_avg = np.array(td.trail_ID_avg)+neo_day
        if neo_day != 0:
            axvline.append(neo_day)
        tot_trails.trail_ID.extend(neo_ID)
        tot_trails.trail_ID_avg.extend(neo_ID_avg)
        neo_day += len(total_route_score_array[i])
        
    if (SB):
        data1 = {
            'trail_ID':tot_trails.trail_ID,
            'route_score':tot_trails.route_score,
            'z_min':tot_trails.z_min,
            'z_max':tot_trails.z_max,
            'z_dif':tot_trails.z_dif,
            'pct_high':tot_trails.pct_high,
            'pct_low':tot_trails.pct_low,
            'SB_peak_frequency':tot_trails.SB_NP_U,
            'SB_zdff_max':tot_trails.SB_zdff_max_U,
            'SB_average_peak_value':tot_trails.SB_avg_PV_U,
            'latency':tot_trails.latency,
            'is_preferred_well':tot_trails.is_preferred_well,
            'day':tot_trails.day
            }
    else:
        data1 = {
            'trail_ID':tot_trails.trail_ID,
            'route_score':tot_trails.route_score,
            'z_min':tot_trails.z_min,
            'z_max':tot_trails.z_max,
            'z_dif':tot_trails.z_dif,
            'pct_high':tot_trails.pct_high,
            'pct_low':tot_trails.pct_low,
            'latency':tot_trails.latency,
            'is_preferred_well':tot_trails.is_preferred_well,
            'day':tot_trails.day
            }
    
    RS = pd.DataFrame(data1)
    if (SB):
        data2 = {
            'route_score_average':tot_trails.route_score_avg,
            'z_max_average':tot_trails.z_max_avg,
            'z_min_average':tot_trails.z_min_avg,
            'SB_peak_frequency':tot_trails.SB_NP,
            'SB_zdff_max':tot_trails.SB_zdff_max,
            'SB_average_peak_value':tot_trails.SB_avg_PV,
            'CB_zdiff_avg':tot_trails.z_dif_avg,
            'day':tot_trails.day_avg
            }
    else:
        data2 = {
            'route_score_average':tot_trails.route_score_avg,
            'z_max_average':tot_trails.z_max_avg,
            'z_min_average':tot_trails.z_min_avg,
            'CB_zdiff_avg':tot_trails.z_dif_avg,
            'day':tot_trails.day_avg
            }
        
    RSA = pd.DataFrame(data2)
    RSA.to_csv(output_folder+mouse_ID+'_Route_Score_Average.csv',index=False)
    
    fig_latency, ax = plt.subplots()
    filtered_df1 = RS[RS['is_preferred_well']]
    filtered_df2 = RS[~RS['is_preferred_well']]
    fdf1 = pd.DataFrame(filtered_df1)
    fdf2 = pd.DataFrame(filtered_df2)
    
    fdf1.to_csv(output_folder+mouse_ID+'_Preferred_Well_Route_Score.csv',index=False)
    fdf2.to_csv(output_folder+mouse_ID+'_Less_Preferred_Well_Route_Score.csv',index=False)
    ax.set_xlabel('Trails')
    ax.set_ylabel('latency', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.plot(filtered_df1['trail_ID'], filtered_df1['latency'], marker='o', linestyle='-', color='b',label='preferred_well')
    ax.plot(filtered_df2['trail_ID'], filtered_df2['latency'], marker='o', linestyle='-', color='g',label='less_preferred_well')
    ax.legend()
    AddVerticalLines(ax, axvline)
    fig_latency.tight_layout()
    fig_latency.savefig(output_folder+'Well_Latency')
    
    fig_pfw = plt.figure(figsize=(10, 10))
    pfwx = fig_pfw.add_subplot(211)
    lpfwx = fig_pfw.add_subplot(212)
    PlotLinearRegression(pfwx, filtered_df1['z_dif'], filtered_df1['route_score'],x_label='z_dif',y_label='route_score')
    PlotLinearRegression(lpfwx, filtered_df2['z_dif'], filtered_df2['route_score'],x_label='z_dif',y_label='route_score')
    pfwx.set_title('z_dif against route score for preferred wells')
    lpfwx.set_title('z_dif against route score for less preferred wells')
    fig_pfw.savefig(output_folder+'preferred_well_tot_plot')
    
    fig = plt.figure(figsize=(10, 30))
    ax1 = fig.add_subplot(711)
    ax2 = fig.add_subplot(712)
    ax3 = fig.add_subplot(713)
    ax4 = fig.add_subplot(714)
    ax5 = fig.add_subplot(715)
    ax6 = fig.add_subplot(716)
    ax7 = fig.add_subplot(717)
    
    PlotLinearRegression(ax1, np.array(tot_trails.route_score), tot_trails.z_min, y_label='z_min')
    PlotLinearRegression(ax2, np.array(tot_trails.route_score), tot_trails.z_max, y_label='z_max')
    PlotLinearRegression(ax3, np.array(tot_trails.route_score_avg), tot_trails.z_min_avg, y_label='z_min_average',x_label='route_score_average')
    PlotLinearRegression(ax4, np.array(tot_trails.route_score_avg), tot_trails.z_max_avg, y_label='z_max_average',x_label='route_score_average')
    PlotLinearRegression(ax5, np.array(tot_trails.route_score), tot_trails.pct_high, y_label='z_percentile_high',x_label='route_score')
    PlotLinearRegression(ax6, np.array(tot_trails.route_score), tot_trails.pct_low, y_label='z_percentile_low',x_label='route_score')
    PlotLinearRegression(ax7, np.array(tot_trails.route_score), tot_trails.z_dif, y_label='z_dif',x_label='route_score')
    
    fig.savefig(output_folder+'Total_CB_Plot')
    
    if (SB):
        fig_SB = plt.figure(figsize=(10, 30))
        SB_ax1 = fig_SB.add_subplot(311)
        SB_ax2 = fig_SB.add_subplot(312)
        SB_ax3 = fig_SB.add_subplot(313)
        PlotLinearRegression(SB_ax1, np.array(tot_trails.route_score_avg), tot_trails.SB_NP, y_label='SB_peak_frequency',x_label='route_score_average')
        PlotLinearRegression(SB_ax2, np.array(tot_trails.route_score_avg), tot_trails.SB_avg_PV, y_label='SB_average_peak_value',x_label='route_score_average')
        PlotLinearRegression(SB_ax3, np.array(tot_trails.route_score_avg), tot_trails.SB_zdff_max, y_label='SB_zdff_max',x_label='route_score_average')
        fig_SB.savefig(output_folder+'Total_SB_Plot')
    
        fig_SB_time = plt.figure(figsize=(15, 30))
        SB_bx1 = fig_SB_time.add_subplot(411)
        SB_bx2 = fig_SB_time.add_subplot(412)
        SB_bx3 = fig_SB_time.add_subplot(413)
        SB_bx4 = fig_SB_time.add_subplot(414)
        SB_bx1.plot(tot_trails.trail_ID_avg, tot_trails.SB_NP, marker='o', linestyle='-', color='b')
        SB_bx2.plot(tot_trails.trail_ID_avg, tot_trails.SB_avg_PV, marker='o', linestyle='-', color='b')
        SB_bx3.plot(tot_trails.trail_ID_avg, tot_trails.SB_zdff_max, marker='o', linestyle='-', color='b')
        SB_bx4.plot(tot_trails.trail_ID_avg, tot_trails.z_dif_avg, marker='o', linestyle='-', color='b')
        
        AddVerticalLines(SB_bx1, axvline)
        AddVerticalLines(SB_bx2, axvline)
        AddVerticalLines(SB_bx3, axvline)
    
        cx1 = SB_bx1.twinx()
        cx1.plot(tot_trails.trail_ID_avg, tot_trails.route_score_avg, marker='o', linestyle='-', color='g')
        cx1.set_ylabel('route_score')
        
        cx2 = SB_bx2.twinx()
        cx2.plot(tot_trails.trail_ID_avg, tot_trails.route_score_avg, marker='o', linestyle='-', color='g')
        cx2.set_ylabel('route_score')
        
        cx3 = SB_bx3.twinx()
        cx3.plot(tot_trails.trail_ID_avg, tot_trails.route_score_avg, marker='o', linestyle='-', color='g')
        cx3.set_ylabel('route_score')
        
        cx4 = SB_bx4.twinx()
        cx4.plot(tot_trails.trail_ID_avg, tot_trails.route_score_avg, marker='o', linestyle='-', color='g')
        cx4.set_ylabel('route_score')
        
        AddVerticalLines(SB_bx4, axvline)
        SB_bx4.set_xlabel('number of trail')
        SB_bx4.set_ylabel('z_dif_CB')
    
        SB_bx1.set_xlabel('number of trail')
        SB_bx2.set_xlabel('number of trail')
        SB_bx3.set_xlabel('number of trail')
        SB_bx1.set_ylabel('peak frequency in SB')
        SB_bx2.set_ylabel('average_peak_value_in_SB')
        SB_bx3.set_ylabel('zdff_max')
        fig_SB_time.savefig(output_folder+'Total_with_time_Plot')
    
#%%
cold_folder = '/Users/zhumingshuai/Desktop/Programming/Photometry/input/1756072/1756072_cold/'
pkl_folder = '/Users/zhumingshuai/Desktop/Programming/Photometry/input/1756072/1756072_pkl/'
output_folder = '/Users/zhumingshuai/Desktop/Programming/Photometry/output/'
PlotRouteScoreGraph(cold_folder, pkl_folder, output_folder,percentile=2.5)



