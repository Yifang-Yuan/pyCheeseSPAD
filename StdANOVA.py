# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:28:29 2024

@author: Mingshuai
"""
import os
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

parameter = {
    'parent_folder':'//cmvm.datastore.ed.ac.uk/cmvm/sbms/users/s2764793/Win7/Desktop/std',
    'key_tag': 'fft'
    
    
    }


def ReadIn (parameter):
    tot_stds = pd.DataFrame()
    min_len = 99999999
    for filename in os.listdir(parameter['parent_folder']):
        if parameter['key_tag'] in filename:
            path = os.path.join(parameter['parent_folder'],filename)
            stds = pd.read_csv(path)
            if stds.shape[0]<min_len:
                min_len = stds.shape[0]
    for filename in os.listdir(parameter['parent_folder']):
        if parameter['key_tag'] in filename:
            path = os.path.join(parameter['parent_folder'],filename)
            stds = pd.read_csv(path)
            tot_stds = pd.concat([tot_stds,stds.iloc[0:min_len]],axis = 1)
    return tot_stds

def ANOVA (parameter,tot_stds):
    groups = [tot_stds[col] for col in tot_stds.columns]
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=tot_stds)
    plt.title('ANOVA test')
    plt.ylabel('fft')
    plt.xlabel('Mouse_ID')
    plt.savefig(os.path.join(parameter['parent_folder'],'FFT_Box_Plot.png'))
    print ('The f_stat is '+str(f_stat))
    print ('The p_value is '+str(p_value))
    return

tot_stds = ReadIn(parameter)
ANOVA(parameter,tot_stds)

            