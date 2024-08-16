#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:44:04 2024

@author: zhumingshuai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import photometry_functions as fp
import pandas as pd
from scipy.stats import linregress
import numpy as np
import os
import re
# import statsmodels.api as sm


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
    
    def __init__ (self,well1_route_score,well2_route_score,z1,z2,day,SB_avg_PV,SB_NP,SB_zdff_max):
        self.well1_route_score = well1_route_score
        self.well2_route_score = well2_route_score
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
        z_1_a = self.z1.iloc[:650]
        z_1_b = self.z1.iloc[650:]
        z_1_a_max = np.percentile(z_1_a,100-pct)
        z_1_b_min = np.percentile(z_1_b,pct)
        self.z_dif_1 = z_1_a_max-z_1_b_min
        z_2_a = self.z2.iloc[:650]
        z_2_b = self.z2.iloc[650:]
        z_2_a_max = np.percentile(z_2_a,100-pct)
        z_2_b_min = np.percentile(z_2_b,pct)
        self.z_dif_2 = z_2_a_max-z_2_b_min
        # if(self.day == 4):
        #     print(z_2_a_max,z_2_b_min)
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
       self.pre_SB_NP = []
       self.pre_SB_avg_PV = []
       self.pre_SB_zdff_max = []
       self.pre_route_score_avg = []
       self.trail_ID = []
       self.z_dif_avg = []
       self.day = []
       self.day_avg = []
    

class truncated_data:
    def __init__ (self,z_min,z_max,route_score,z_min_avg,z_max_avg,route_score_avg,pct_high,pct_low,z_dif,SB_avg_PV,SB_NP,SB_zdff_max,pre_SB_avg_PV,pre_SB_NP,pre_SB_zdff_max,pre_route_score_avg,trail_ID,z_dif_avg,day_ID,day_ID_avg):
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
        self.pre_SB_NP = pre_SB_NP
        self.pre_SB_avg_PV = pre_SB_avg_PV
        self.pre_SB_zdff_max = pre_SB_zdff_max
        self.pre_route_score_avg = pre_route_score_avg
        self.trail_ID = trail_ID
        self.z_dif_avg = z_dif_avg
        self.day_ID = day_ID
        self.day_ID_avg = day_ID_avg
    
def ReadRouteScore (cold_folder, cold_file,pickle_folder,pickle_file,day,percentile):
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
                print(filename)
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
        if (route_score_input['firstwellreached'][i]==2):
            z1,z2 = z2,z1
        single_trail_score = trail_route_score(well1_route_score,well2_route_score,z1,z2,day,SB_avg_PV,SB_NP,SB_zdff_max)
        single_trail_score.Calculate(percentile)
        route_score_array.append(single_trail_score)    
        
    return route_score_array


def Plot_Single_RS (route_score_array,day,output_folder):
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
    pre_SB_NP = []
    pre_SB_avg_PV = []
    pre_SB_zdff_max = []
    pre_route_score_avg = []
    trail_ID = []
    z_dif_avg = []
    day_ID = []
    day_ID_avg = []
    filename_prefix = 'Route_Score_Plot_Day'
    for i in range (len(route_score_array)):
        if ((not np.isnan(route_score_array[i].well1_route_score)) and not(np.isnan(route_score_array[i].z_dif_1))):
            z_min.append(route_score_array[i].z_min_1)
            z_max.append(route_score_array[i].z_max_1)
            pct_high.append(route_score_array[i].pct_h1)
            pct_low.append(route_score_array[i].pct_l1)
            z_dif.append(route_score_array[i].z_dif_1)
            day_ID.append(day)
            # if (np.isnan(route_score_array[i].z_dif_1)):
                # print('error!')
                # print(i,day)
                # print(route_score_array[i].z1)
            route_score.append(route_score_array[i].well1_route_score)
        if (not np.isnan(route_score_array[i].well2_route_score) and not(np.isnan(route_score_array[i].z_dif_2))):
            z_min.append(route_score_array[i].z_min_2)
            z_max.append(route_score_array[i].z_max_2)
            pct_high.append(route_score_array[i].pct_h2)
            pct_low.append(route_score_array[i].pct_l2)
            route_score.append(route_score_array[i].well2_route_score)
            z_dif.append(route_score_array[i].z_dif_2)
            day_ID.append(day)
            # if (np.isnan(route_score_array[i].z_dif_2)):
            #     print('error!')
            #     print(route_score_array[i].z1)
            #     print(route_score_array[i].z2)
        if ((not np.isnan(route_score_array[i].well2_route_score)) and (not np.isnan(route_score_array[i].well1_route_score)) and not(np.isnan(route_score_array[i].z_dif_1)) and not(np.isnan(route_score_array[i].z_dif_2))):
            z_max_avg.append(route_score_array[i].z_max_avg)
            z_min_avg.append(route_score_array[i].z_min_avg)
            SB_NP.append(route_score_array[i].SB_NP)
            SB_avg_PV.append(route_score_array[i].SB_avg_PV)
            SB_zdff_max.append(route_score_array[i].SB_zdff_max)
            route_score_avg.append(route_score_array[i].route_score_avg)
            z_dif_avg.append((route_score_array[i].z_dif_2+route_score_array[i].z_dif_1)/2)
            trail_ID.append(i)
            day_ID_avg.append(day)
            if(i!=len(route_score_array)-1):
                pre_SB_NP.append(route_score_array[i+1].SB_NP)
                pre_SB_avg_PV.append(route_score_array[i+1].SB_avg_PV)
                pre_SB_zdff_max.append(route_score_array[i+1].SB_zdff_max)
                pre_route_score_avg.append(route_score_array[i].route_score_avg)
            
    fig = plt.figure(figsize=(10, 30))
    ax1 = fig.add_subplot(711)
    ax2 = fig.add_subplot(712)
    ax3 = fig.add_subplot(713)
    ax4 = fig.add_subplot(714)
    ax5 = fig.add_subplot(715)
    ax6 = fig.add_subplot(716)
    ax7 = fig.add_subplot(717)
    PlotLinearRegression(ax1, np.array(route_score), z_min, y_label='z_min')
    PlotLinearRegression(ax2, np.array(route_score), z_max, y_label='z_max')
    PlotLinearRegression(ax3, np.array(route_score_avg), z_min_avg, y_label='z_min_average',x_label='route_score_average')
    PlotLinearRegression(ax4, np.array(route_score_avg), z_max_avg, y_label='z_max_average',x_label='route_score_average')
    PlotLinearRegression(ax5, np.array(route_score), pct_high, y_label='z_percentile_high',x_label='route_score')
    PlotLinearRegression(ax6, np.array(route_score), pct_low, y_label='z_percentile_low',x_label='route_score')
    PlotLinearRegression(ax7, np.array(route_score), z_dif, y_label='z_dif',x_label='route_score')
    
    fig_SB = plt.figure(figsize=(10, 30))
    SB_ax1 = fig_SB.add_subplot(611)
    SB_ax2 = fig_SB.add_subplot(612)
    SB_ax3 = fig_SB.add_subplot(613)
    SB_ax4 = fig_SB.add_subplot(614)
    SB_ax5 = fig_SB.add_subplot(615)
    SB_ax6 = fig_SB.add_subplot(616)
    PlotLinearRegression(SB_ax1, np.array(route_score_avg), SB_NP, y_label='SB_peak_frequency',x_label='route_score_average')
    PlotLinearRegression(SB_ax2, np.array(route_score_avg), SB_avg_PV, y_label='SB_average_peak_value',x_label='route_score_average')
    PlotLinearRegression(SB_ax3, np.array(route_score_avg), SB_zdff_max, y_label='SB_zdff_max',x_label='route_score_average')
    PlotLinearRegression(SB_ax4, np.array(pre_route_score_avg), pre_SB_NP, y_label='SB_peak_frequency(pre)',x_label='route_score_average')
    PlotLinearRegression(SB_ax5, np.array(pre_route_score_avg), pre_SB_avg_PV, y_label='SB_average_peak_value(pre)',x_label='route_score_average')
    PlotLinearRegression(SB_ax6, np.array(pre_route_score_avg), pre_SB_zdff_max, y_label='SB_zdff_max(pre)',x_label='route_score_average')
    
    fig_SB_time = plt.figure(figsize=(10, 30))
    SB_bx1 = fig_SB_time.add_subplot(311)
    SB_bx2 = fig_SB_time.add_subplot(312)
    SB_bx3 = fig_SB_time.add_subplot(313)
    SB_bx1.plot(trail_ID, SB_NP, marker='o', linestyle='-', color='b')
    SB_bx2.plot(trail_ID, SB_avg_PV, marker='o', linestyle='-', color='b')
    SB_bx3.plot(trail_ID, SB_zdff_max, marker='o', linestyle='-', color='b')
    SB_bx1.set_xlabel('number of trail')
    SB_bx2.set_xlabel('number of trail')
    SB_bx3.set_xlabel('number of trail')
    SB_bx1.set_ylabel('peak frequency in SB')
    SB_bx2.set_ylabel('average_peak_value_in_SB')
    SB_bx3.set_ylabel('zdff_max')
    fig_SB_time.savefig(output_folder+'SB_with_time_'+filename_prefix+str(day))
    
    
    fig.savefig(output_folder+'CB_'+filename_prefix+str(day))
    fig_SB.savefig(output_folder+'SB_'+filename_prefix+str(day))
    # if(day==2):
    #     print(pct_high)
    return truncated_data(z_min, z_max, route_score, z_min_avg, z_max_avg, route_score_avg,pct_high,pct_low,z_dif,SB_avg_PV,SB_NP,SB_zdff_max,pre_SB_avg_PV,pre_SB_NP,pre_SB_zdff_max,pre_route_score_avg,trail_ID,z_dif_avg,day_ID,day_ID_avg)

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

def PlotRouteScoreGraph (cold_folder, cold_filename, pickle_folder, pickle_filename,output_folder,percentile=5):
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
    for filename in os.listdir(cold_folder):
        if filename.endswith('.xlsx'):
            if 'Day' in filename:
                day = re.findall(r'\d+', filename.split('Day')[1])
                day = day[0]
                day = int(day)
                files[day-1][0] = filename
    total_route_score_array = []
    
    for i in range (len(files)):
        print(files[i][0])
        print(files[i][1])
        route_score_array = ReadRouteScore(cold_folder, files[i][0],pickle_folder,files[i][1],i+1,percentile=percentile)
        total_route_score_array.append(route_score_array)
    
    single_plot_output = output_folder+'Single_Day_Plot/'
    if not os.path.exists(single_plot_output):
        os.makedirs(single_plot_output)
    
 #integrate data
    axvline = []
    neo_day = 0
    for i in range (len(total_route_score_array)):  
        td = Plot_Single_RS(total_route_score_array[i], i+1 , single_plot_output)
        tot_trails.z_min.extend(td.z_min)
        tot_trails.z_max.extend(td.z_max)
        tot_trails.route_score.extend(td.route_score)
        tot_trails.z_min_avg.extend(td.z_min_avg)
        tot_trails.z_max_avg.extend(td.z_max_avg)
        tot_trails.route_score_avg.extend(td.route_score_avg)
        tot_trails.pct_high.extend(td.pct_high)
        tot_trails.pct_low.extend(td.pct_low)
        tot_trails.z_dif.extend(td.z_dif)
        tot_trails.SB_NP.extend(td.SB_NP)
        tot_trails.SB_avg_PV.extend(td.SB_avg_PV)
        tot_trails.SB_zdff_max.extend(td.SB_zdff_max)
        tot_trails.pre_SB_NP.extend(td.pre_SB_NP)
        tot_trails.pre_SB_avg_PV.extend(td.pre_SB_avg_PV)
        tot_trails.pre_SB_zdff_max.extend(td.pre_SB_zdff_max)
        tot_trails.pre_route_score_avg.extend(td.pre_route_score_avg)
        tot_trails.z_dif_avg.extend(td.z_dif_avg)
        tot_trails.day.extend(td.day_ID)
        tot_trails.day_avg.extend(td.day_ID_avg)
        neo_ID = np.array(td.trail_ID)+neo_day
        if neo_day != 0:
            axvline.append(neo_day)
        tot_trails.trail_ID.extend(neo_ID)
        neo_day += len(total_route_score_array[i])
        
    
    data1 = {
        'route_score':tot_trails.route_score,
        'z_min':tot_trails.z_min,
        'z_max':tot_trails.z_max,
        'z_dif':tot_trails.z_dif,
        'pct_high':tot_trails.pct_high,
        'pct_low':tot_trails.pct_low,
        'day':tot_trails.day
        }
    RS = pd.DataFrame(data1)
    
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
    RSA = pd.DataFrame(data2)
    RS.to_csv(output_folder+'Route_Score.csv',index=False)
    RSA.to_csv(output_folder+'Route_Score_Average.csv',index=False)
    
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
    
    fig_SB = plt.figure(figsize=(10, 30))
    SB_ax1 = fig_SB.add_subplot(611)
    SB_ax2 = fig_SB.add_subplot(612)
    SB_ax3 = fig_SB.add_subplot(613)
    SB_ax4 = fig_SB.add_subplot(614)
    SB_ax5 = fig_SB.add_subplot(615)
    SB_ax6 = fig_SB.add_subplot(616)
    PlotLinearRegression(SB_ax1, np.array(tot_trails.route_score_avg), tot_trails.SB_NP, y_label='SB_peak_frequency',x_label='route_score_average')
    PlotLinearRegression(SB_ax2, np.array(tot_trails.route_score_avg), tot_trails.SB_avg_PV, y_label='SB_average_peak_value',x_label='route_score_average')
    PlotLinearRegression(SB_ax3, np.array(tot_trails.route_score_avg), tot_trails.SB_zdff_max, y_label='SB_zdff_max',x_label='route_score_average')
    PlotLinearRegression(SB_ax4, np.array(tot_trails.pre_route_score_avg), tot_trails.pre_SB_NP, y_label='SB_num_peaks(pre)',x_label='route_score_average')
    PlotLinearRegression(SB_ax5, np.array(tot_trails.pre_route_score_avg), tot_trails.pre_SB_avg_PV, y_label='SB_average_peak_value(pre)',x_label='route_score_average')
    PlotLinearRegression(SB_ax6, np.array(tot_trails.pre_route_score_avg), tot_trails.pre_SB_zdff_max, y_label='SB_zdff_max(pre)',x_label='route_score_average')
    fig_SB.savefig(output_folder+'Total_SB_Plot')

    fig_SB_time = plt.figure(figsize=(10, 30))
    SB_bx1 = fig_SB_time.add_subplot(411)
    SB_bx2 = fig_SB_time.add_subplot(412)
    SB_bx3 = fig_SB_time.add_subplot(413)
    SB_bx4 = fig_SB_time.add_subplot(414)
    SB_bx1.plot(tot_trails.trail_ID, tot_trails.SB_NP, marker='o', linestyle='-', color='b')
    SB_bx2.plot(tot_trails.trail_ID, tot_trails.SB_avg_PV, marker='o', linestyle='-', color='b')
    SB_bx3.plot(tot_trails.trail_ID, tot_trails.SB_zdff_max, marker='o', linestyle='-', color='b')
    AddVerticalLines(SB_bx1, axvline)
    AddVerticalLines(SB_bx2, axvline)
    AddVerticalLines(SB_bx3, axvline)
    
    cx1 = SB_bx1.twinx()
    cx1.plot(tot_trails.trail_ID, tot_trails.route_score_avg, marker='o', linestyle='-', color='g')
    cx1.set_ylabel('route_score')
    
    cx2 = SB_bx4.twinx()
    SB_bx4.plot(tot_trails.trail_ID, tot_trails.z_dif_avg, marker='o', linestyle='-', color='b')
    cx2.plot(tot_trails.trail_ID, tot_trails.route_score_avg, marker='o', linestyle='-', color='g')
    cx2.set_ylabel('route_score')
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
    
    
    # X = sm.add_constant(tot_trails.route_score_avg)
    # model = sm.OLS(tot_trails.SB_NP, X).fit()
    # print(model.summary())

    
#%%
COLD_filename = 'Training_Data_Day1.xlsx'   #not using though
pickle_filename = '1756072_Day1_5sec_win_traces.pkl'  #not using though


output_folder='F:/CB_EC5aFibre_1756072/correlationResults/'
pickle_folder='F:/CB_EC5aFibre_1756072/results/'
COLD_folder='F:/CB_EC5aFibre_1756072/COLD_folder/'
animalID='1756072'
PlotRouteScoreGraph(COLD_folder, COLD_filename, pickle_folder, pickle_filename, output_folder,percentile=5)



