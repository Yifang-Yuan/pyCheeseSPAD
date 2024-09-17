# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:44:51 2024

@author: Mingshuai
"""
import os
import pandas as pd
input_df = {
    'parent_folder' : 'E:/workingfolder/Group E/6535/6535_Cold_folder/',
    'neo_fr' : 16,
    'pre_fr' : 24,
    'key_tag': 'Training_Data',
    'modify_columns': ['startingtime_s','well1time_s','well2time_s','leftfirstwell_s','latencybetweenwells_s','trialtime_s']
    }


def Modification(x,input_df):
    return x*input_df['pre_fr']/input_df['neo_fr']

def ReadIn (input_df):
    for filename in os.listdir(input_df['parent_folder']):
        if filename == ('hachimi.txt'):
            print('Warning, the folder has already being modified')
            return
    for filename in os.listdir(input_df['parent_folder']):
        if not (filename.endswith('.xlsx') and input_df['key_tag'] in filename):
            print('error! there are something else in the folder')
        else:
            file_path = os.path.join(input_df['parent_folder'],filename)
            file = pd.read_excel(file_path)
            for col in input_df['modify_columns']:
                file[col] = Modification(file[col],input_df)
            
            file.to_excel(file_path, index=False)
            print(filename+'modified!')
    hachimi = os.path.join(input_df['parent_folder'],'hachimi.txt')
    with open(hachimi, 'w') as file:
        file.write('manba out!')
        
ReadIn (input_df)
