#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:21:56 2024

@author: Mingshuai Zhu
"""

import pandas as pd
import openpyxl
import re
import os

input_folder = '/Volumes/YifangExp/Mingshuai/Group A/output/1079&1081/'
input_file = 'Training Data.xlsx'
file_name_column = 'name'
output_folder_prefix = '/Volumes/YifangExp/Mingshuai/Group A/output/1079&1081/'
output_file_prefix = 'Training_Data_Day'

def ReadTrainingData (input_folder,input_file,file_name_column):
    training_data = pd.read_excel(input_folder+input_file)
    return training_data

def ObtainHeader (training_data):
    header = training_data.columns.tolist()
    return header

class SingleTrail:
    def __init__(self,mouse_ID,day,ID,content):
        self.mouse_ID = mouse_ID
        self.day = day
        self.ID = ID
        self.content = content
        
class Mouse:
    def __init__(self,ID,max_day):
        self.ID = ID
        self.max_day = max_day
        self.multiple_trails = []

    def AddTrail(self,trail):
        self.multiple_trails.append(trail)

    def SortTrails(self):
        self.multiple_trails.sort(key=lambda x:x.ID)
        
def ReadData (training_data, row_index):
    name = training_data.iloc[row_index,0]
    probe = False
    if 'day' in name:
        split_name = name.split('day')
    elif 'Day' in name:
        split_name = name.split('Day')
    elif 'Probe' in name:
        probe = True
        split_name = name.split('Probe')
    elif 'probe' in name:
        probe = True
        split_name = name.split('probe')
    else:
        print ('Error! Day not found!')
    
    #obtain all numbers in the substring after 'day'
    numbers_in_name = re.findall(r'\d+', split_name[1])
    mouse_ID = re.findall(r'\d+', split_name[0])[0]
    if(probe):
        day = -1
        ID = int(numbers_in_name[0])
    else:
        day = int(numbers_in_name[0])
        ID = int(numbers_in_name[1])
    return SingleTrail(mouse_ID, day, ID, training_data.iloc[row_index].tolist())

def OutputTrainingData (multiple_trails,day,header,output_folder):
    if (day == -1):
        output_file_name = 'Training_Data_Probe.xlsx'
    else:
        output_file_name = output_file_prefix+str(day)+'.xlsx'
    
    output_file_path = output_folder+output_file_name
    output_workbook = openpyxl.Workbook()
    active_workbook = output_workbook.active
    active_workbook.append(header)
    
    for i in range (len(multiple_trails)):
        if (multiple_trails[i].day == day):
            active_workbook.append(multiple_trails[i].content)
    output_workbook.save(output_file_path)
    return

def MouseExist (mouse_ID, mouse_list):
    x = -1
    for i in range (len(mouse_list)):
        if (mouse_list[i].ID == mouse_ID):
            x = i
            break
    return x

#%%
"""
Main function
"""

training_data = ReadTrainingData(input_folder, input_file, file_name_column)

header = ObtainHeader(training_data)

mouse_list = []

for i in range (training_data.shape[0]):
    single_trail = ReadData(training_data, i)
    mouse_index = MouseExist(single_trail.mouse_ID, mouse_list)
    if (mouse_index!=-1):
        mouse_list[mouse_index].AddTrail(single_trail)
    else:
        mouse_list.append(Mouse(single_trail.mouse_ID,-1))
        mouse_list[mouse_index].AddTrail(single_trail)
    if (single_trail.day>mouse_list[mouse_index].max_day):
        mouse_list[mouse_index].max_day = single_trail.day

for i in range(len(mouse_list)):
    mouse_list[i].SortTrails()
    output_folder = output_folder_prefix+str(mouse_list[i].ID)+'/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for j in range (1,mouse_list[i].max_day+1):
        OutputTrainingData(mouse_list[i].multiple_trails, j, header,output_folder)
    OutputTrainingData(mouse_list[i].multiple_trails, -1, header,output_folder)
