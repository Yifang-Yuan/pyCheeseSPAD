#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:00:30 2024

@author: zhumingshuai
"""
import os
folder_path = '/Volumes/YifangExp/Mingshuai/workingfolder/Group A/Group A (cue)/'
old_substring = '1079'
new_substring = '1756072'
for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in filenames:
        if old_substring in filename:
            # Construct the full old and new file paths
            old_file_path = os.path.join(dirpath, filename)
            new_filename = filename.replace(old_substring, new_substring)
            new_file_path = os.path.join(dirpath, new_filename)
            
            try:
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {old_file_path} -> {new_file_path}')
            except Exception as e:
                print(f'Error renaming {old_file_path}: {e}')