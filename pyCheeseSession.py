# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 12:28:54 2024
@author: Yifang

This is a Class to process cheeseboard-photometry data of 1 session.
1-session: multiple trials that performed in one day or one group.
This code is usually used to process trials from one day and then compare across days.
However, you can also put trials from multiple days to a single folder to creates a Session, and process with this code.
 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import photometry_functions as fp
import os
import re
import scipy.signal as signal
import pickle

class pyCheeseSession:
    def __init__(self, pyFolder, bonsai_folder,CheeseFolder,COLD_filename,save_folder,animalID='mice1',SessionID='Day1',pySBFolder=None):
        '''
        Parameters
        ----------
        SessionPath : path to save data for a single recording trial with ephys and optical data. 
        IsTracking: 
            whether to include animal position tracking data, not neccessary for sleeping sessions.
        read_aligned_data_from_file: 
            False if it is the first time you analyse this trial of data, 
            once you have removed noises, run the single-trial analysis, saved the .pkl file, 
            set it to true to read aligned data directly.
        '''
        'Define photometry recording sampling rate and Camera frame rate '
        self.pyFs=130
        self.CamFs=24
        self.pyFolder=pyFolder
        self.bonsai_folder = bonsai_folder
        self.CheeseFolder=CheeseFolder
        self.animalID=animalID
        self.SessionID=SessionID
        self.COLD_resultfilename=COLD_filename
        self.save_folder=save_folder
        self.sort_py_files(self.pyFolder)
        self.form_pyCheesed_data_to_pandas()
        self.result_path = os.path.join(self.save_folder, 'results')
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if not pySBFolder==None:
            self.pySBFolder=pySBFolder
            self.sort_py_files(self.pySBFolder)
            
        
    def sort_py_files (self,folder):
        original_pattern = re.compile(r'(\w+)-(\d{4}-\d{2}-\d{2}-\d{6})(\.csv)?$')
        renamed_pattern = re.compile(r'py_(\w+)-(\d{4}-\d{2}-\d{2}-\d{6})_(\d+)\.csv$')
        partially_renamed_pattern = re.compile(r'(\w+)-(\d{4}-\d{2}-\d{2}-\d{6})_(\d+)\.csv$')
        # List all files in the directory that match the patterns
        files = [f for f in os.listdir(folder) if original_pattern.match(f) or renamed_pattern.match(f) or partially_renamed_pattern.match(f)]
        
        # Extract timestamps and filenames into a list of tuples
        files_with_timestamps = [(original_pattern.match(f).group(2), f) if original_pattern.match(f) else 
                                 (renamed_pattern.match(f).group(2), f) if renamed_pattern.match(f) else 
                                 (partially_renamed_pattern.match(f).group(2), f) for f in files]
        # Check if files are already sorted and renamed
        already_sorted_and_renamed = all(
            renamed_pattern.match(f) and int(renamed_pattern.match(f).group(3)) == i for i, (timestamp, f) in enumerate(sorted(files_with_timestamps, key=lambda x: x[0]))
        )

        # If not sorted and renamed, sort files based on the timestamp part
        if not already_sorted_and_renamed:
            files_with_timestamps.sort()
            # Rename files
            for index, (timestamp, filename) in enumerate(files_with_timestamps):
                match = original_pattern.match(filename) if original_pattern.match(filename) else partially_renamed_pattern.match(filename)
                new_filename = f"py_{match.group(1)}-{match.group(2)}_{index}.csv"
                
                # Construct full file paths
                old_filepath = os.path.join(folder, filename)
                new_filepath = os.path.join(folder, new_filename)
                
                # Rename the file
                os.rename(old_filepath, new_filepath)
                print(f'Renamed "{filename}" to "{new_filename}"')

        else:
            print("Files are already sorted and renamed correctly.")
        return -1
    
    def form_pyCheesed_data_to_pandas(self):
        py_target_string='py'
        sync_target_string='sync'
        
        files = os.listdir(self.pyFolder)
        bonsai_files = os.listdir(self.bonsai_folder)
        filtered_files = [file for file in files if py_target_string in file]  
        cheeaseboard_session_data=fp.read_cheeseboard_from_COLD (self.CheeseFolder, self.COLD_resultfilename)
        photometry_df = pd.DataFrame([])
        cheese_df=pd.DataFrame([])
        for file in filtered_files:
            # Extract the last number from the file name
            match = re.search(r'_(\d+)\.csv$', file)
            if match:
                target_index= match.group(1)
            # Read the CSV file with photometry read
            raw_signal,raw_reference,Cam_Sync=fp.read_photometry_data (self.pyFolder, file, readCamSync=True,plot=False,sampling_rate=130)
            zdFF = fp.get_zdFF(raw_reference,raw_signal,smooth_win=10,remove=0,lambd=5e4,porder=1,itermax=50) 
            filtered_files_sync = [file for file in bonsai_files if sync_target_string in file]
            for Sync_file in filtered_files_sync:
                match_sync = re.search(r'sync_(\d+)\.csv$', Sync_file)
                if match_sync:
                    sync_index= match_sync.group(1)
                if sync_index == target_index:
                    CamSync_LED=fp.read_Bonsai_Sync (self.bonsai_folder,Sync_file,plot=False)
                    zscore_sync,Sync_index_inCam,Sync_Start_time=fp.sync_photometry_Cam(zdFF,Cam_Sync,CamSync_LED,CamFs=self.CamFs)
                    zscore_series=pd.Series(zscore_sync)
                    zscore_series=zscore_series.reset_index(drop=True)
                    Sync_Start_time_series=pd.Series(Sync_Start_time)
                    entertime, well1time,well2time,leftfirstwell_time=fp.adjust_time_to_photometry(cheeaseboard_session_data,
                                                                                                   int(target_index),Sync_Start_time)
                    if np.isnan(well1time):
                        real_well1time = well2time
                        real_well2time =well1time
                    elif np.isnan(well2time):
                        real_well1time = well1time
                        real_well2time=well2time
                    else:
                        # Both numbers are real, so find the larger one
                        real_well1time = np.minimum(well1time, well2time)
                        real_well2time=np.maximum(well1time, well2time)
                    photometry_df['pyData'+target_index]=zscore_series
                    cheese_df['SyncStartTimeInCam'+target_index]=Sync_Start_time_series
                    cheese_df['entertime'+target_index]=pd.Series(entertime)
                    cheese_df['well1time'+target_index]=pd.Series(real_well1time)
                    cheese_df['well2time'+target_index]=pd.Series(real_well2time)
                    cheese_df['leftwell1time'+target_index]=pd.Series(leftfirstwell_time)
        self.photometry_df=photometry_df
        filename=self.animalID+'_'+self.SessionID+'_'+'full_trial_zscore.pkl'
        self.photometry_df.to_pickle(os.path.join(self.result_path, filename))
        self.cheese_df=cheese_df
        return self.photometry_df,self.cheese_df
        
    def Plot_multiple_PETH_different_window(self,before_window,after_window):
        reward_event_window_traces = pd.DataFrame([])
        selected_columns = [col_name for col_name in self.photometry_df.columns if col_name.startswith('pyData') and col_name[6:].isdigit()]
        #('selected_columns:',selected_columns)
        column_numbers = [int(col_name.replace('pyData', '')) for col_name in selected_columns]  
        event_time = 0 
        '''This is to make a figure and plot all PETH traces on the same figure'''
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        for col_num in column_numbers:
            column_name = f'pyData{col_num}'
            column_photometry_data = self.photometry_df[column_name]
            column_well1time=self.cheese_df[f'well1time{col_num}'][0]
            if not np.isnan(column_well1time).any():
                start_idx=int((column_well1time-before_window)*self.pyFs)
                end_idx=int((column_well1time+after_window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                reward_event_window_traces[f'pyData{col_num}'+'_1']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax1, self.photometry_df[f'pyData{col_num}'],centre_time=
                                 self.cheese_df[f'well1time{col_num}'][0], before_window=before_window,
                                 after_window=after_window, fs=self.pyFs,color='green',Label=f'Trace{col_num+1} Well1time')
            else:
                length=(before_window+after_window)*self.pyFs
                reward_event_window_traces[f'pyData{col_num}'+'_1']=pd.Series(np.nan, index=range(length))
            column_well2time=self.cheese_df[f'well2time{col_num}'][0]
            if not np.isnan(column_well2time).any():
                start_idx=int((column_well2time-before_window)*self.pyFs)
                end_idx=int((column_well2time+after_window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                reward_event_window_traces[f'pyData{col_num}'+'_2']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax2, self.photometry_df[f'pyData{col_num}'],centre_time=
                                 self.cheese_df[f'well2time{col_num}'][0], before_window=before_window,
                                 after_window=after_window, fs=self.pyFs,color='red',Label=f'Trace{col_num+1} Well2time')
            else:
                length=(before_window+after_window)*self.pyFs
                reward_event_window_traces[f'pyData{col_num}'+'_2']=pd.Series(np.nan, index=range(length))
        ax1.axvline(x=event_time, color='red', linestyle='--', label='Event Time') 
        ax2.axvline(x=event_time, color='red', linestyle='--', label='Event Time') 
        ax1.set_title("1st reached reward")
        ax2.set_title("2nd reached reward")
        plt.xlabel('Time (second)')
        plt.ylabel('Value')
        fig.suptitle('Single traces while reaching well1 (green) and well2 (red) '+self.animalID)
        plt.tight_layout()
        plt.show()
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+'_'+str(before_window)+'sec_window_reward_single.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        
        'separate well 1 and well 2 and plot average traces'
        filtered_columns = [col for col in reward_event_window_traces.columns if col.endswith('_1')]
        Well1_PETH = reward_event_window_traces[filtered_columns]
        filtered_columns = [col for col in reward_event_window_traces.columns if col.endswith('_2')]
        Well2_PETH = reward_event_window_traces[filtered_columns]
        '''plot'''
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fp.Plot_mean_With_CI_PSTH(Well1_PETH, before_window, after_window, self.animalID, meancolor='green', stdcolor='lightgreen', ax=ax1)
        fp.Plot_mean_With_CI_PSTH(Well2_PETH, before_window, after_window, self.animalID, meancolor='red', stdcolor='lightcoral', ax=ax2)
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+'_'+str(before_window)+'sec_window_reward_average.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        'plot average trace for both rewards'
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        fp.Plot_mean_With_CI_PSTH(reward_event_window_traces, before_window, after_window, self.animalID, meancolor='blue', stdcolor='lightblue', ax=ax)
        ax.legend()
        plt.show()
        output_path = os.path.join(self.result_path, self.animalID+'_'+self.SessionID+'_'+str(before_window)+'sec_window_reward_mean_std.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        '''save the pkl file for the PETH data with half window time specified'''
        filename=self.animalID+'_'+self.SessionID+'_'+str(before_window)+'sec_win_traces.pkl'
        self.reward_event_window_traces=reward_event_window_traces
        self.reward_event_window_traces.to_pickle(os.path.join(self.result_path, filename))
        return self.reward_event_window_traces
    
    def Event_time_single_side(self,window):
        event_time_photometry_trace = pd.DataFrame([])
        selected_columns = [col_name for col_name in self.photometry_df.columns if col_name.startswith('pyData') and col_name[6:].isdigit()]
        print ('selected_columns:',selected_columns)
        column_numbers = [int(col_name.replace('pyData', '')) for col_name in selected_columns]  
        '''This is to make a figure and plot all PETH traces on the same figure'''

        fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2, figsize=(10, 10))
        for col_num in column_numbers:
            column_name = f'pyData{col_num}'
            column_photometry_data = self.photometry_df[column_name]
            
            column_entertime=self.cheese_df[f'entertime{col_num}'][0]
            print ('column_entertime--',f'entertime{col_num}--',column_entertime)
            if not np.isnan(column_entertime).any():
                start_idx=int(column_entertime*self.pyFs)
                end_idx=int((column_entertime+window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                #print ('para_ENTER_photometry_data',para_event_photometry_data)
                event_time_photometry_trace[f'pyData{col_num}'+'_enter']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax1, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'entertime{col_num}'][0], before_window=0,
                                  after_window=window, fs=self.pyFs,color='grey',Label=f'Trace{col_num+1} enter time')
            else:
                length=(window)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_enter']=pd.Series(np.nan, index=range(length))
            
            
            column_well1time=self.cheese_df[f'well1time{col_num}'][0]
            if not np.isnan(column_well1time).any():
                start_idx=int((column_well1time-window)*self.pyFs)
                end_idx=int((column_well1time)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                event_time_photometry_trace[f'pyData{col_num}'+'_beforewell1']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax2, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'well1time{col_num}'][0], before_window=window,
                                  after_window=0, fs=self.pyFs,color='green',Label=f'Trace{col_num+1} before well1')
            else:
                length=(window)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_beforewell1']=pd.Series(np.nan, index=range(length))
           
            column_leftwll1time=self.cheese_df[f'leftwell1time{col_num}'][0]
            if not np.isnan(column_leftwll1time).any():
                start_idx=int(column_leftwll1time*self.pyFs)
                end_idx=int((column_leftwll1time+window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                event_time_photometry_trace[f'pyData{col_num}'+'_leftwell1']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax3, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'leftwell1time{col_num}'][0], before_window=0,
                                  after_window=window, fs=self.pyFs,color='green',Label=f'Trace{col_num+1} enter time')
            else:
                length=(window)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_leftwell1']=pd.Series(np.nan, index=range(length))   
           
            column_well2time=self.cheese_df[f'well2time{col_num}'][0]
            if not np.isnan(column_well2time).any():
                start_idx=int((column_well2time-window)*self.pyFs)
                end_idx=int((column_well2time)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                event_time_photometry_trace[f'pyData{col_num}'+'_beforewell2']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax4, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'well2time{col_num}'][0], before_window=window,
                                  after_window=0, fs=self.pyFs,color='red',Label=f'Trace{col_num+1} before Well2')
            else:
                length=(window)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_beforewell2']=pd.Series(np.nan, index=range(length))
        plt.xlabel('Time (second)')
        plt.ylabel('Value')
        fig.suptitle('Event time traces '+self.animalID)
        plt.tight_layout()
        plt.show()
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+str(window)+'sec_window_event_traces.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        
        'filtered each event and plot average'
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_enter')]
        entertime_PETH = event_time_photometry_trace[filtered_columns]
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_beforewell1')]
        before_well1_PETH= event_time_photometry_trace[filtered_columns]
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_leftwell1')]
        left_well1_PETH= event_time_photometry_trace[filtered_columns]
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_beforewell2')]
        before_well2_PETH= event_time_photometry_trace[filtered_columns]
                                                                        
        fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2, figsize=(10, 10))
        fp.Plot_mean_With_CI_PSTH(entertime_PETH, 0, window, self.animalID, meancolor='grey', stdcolor='lightgrey', ax=ax1)
        fp.Plot_mean_With_CI_PSTH(before_well1_PETH, window, 0, self.animalID, meancolor='green', stdcolor='lightgreen', ax=ax2)
        fp.Plot_mean_With_CI_PSTH(left_well1_PETH, 0, window, self.animalID, meancolor='green', stdcolor='lightgreen', ax=ax3)
        fp.Plot_mean_With_CI_PSTH(before_well2_PETH, window, 0, self.animalID, meancolor='red', stdcolor='lightcoral', ax=ax4)
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+'_'+str(window)+'sec_window_event_average.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)

        '''save the pkl file for the PETH data with half window time specified'''
        filename=self.animalID+'_'+self.SessionID+'_'+str(window)+'sec_half_window_event_traces.pkl'
        self.event_time_photometry_trace=event_time_photometry_trace
        self.event_time_photometry_trace.to_pickle(os.path.join(self.result_path, filename))
        return self.event_time_photometry_trace
    
    def Event_time_two_sides(self,window):
        event_time_photometry_trace = pd.DataFrame([])
        selected_columns = [col_name for col_name in self.photometry_df.columns if col_name.startswith('pyData') and col_name[6:].isdigit()]
        print ('selected_columns:',selected_columns)
        column_numbers = [int(col_name.replace('pyData', '')) for col_name in selected_columns]  
        '''This is to make a figure and plot all PETH traces on the same figure'''

        fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2, figsize=(10, 10))
        for col_num in column_numbers:
            column_name = f'pyData{col_num}'
            column_photometry_data = self.photometry_df[column_name]
            
            column_entertime=self.cheese_df[f'entertime{col_num}'][0]
            print ('column_entertime--',f'entertime{col_num}--',column_entertime)
            if not np.isnan(column_entertime).any():
                start_idx=int((column_entertime-window)*self.pyFs)
                end_idx=int((column_entertime+window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                #print ('para_ENTER_photometry_data',para_event_photometry_data)
                event_time_photometry_trace[f'pyData{col_num}'+'_enter']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax1, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'entertime{col_num}'][0], before_window=window,
                                  after_window=window, fs=self.pyFs,color='grey',Label=f'Trace{col_num+1} enter time')
            else:
                length=(window*2)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_enter']=pd.Series(np.nan, index=range(length))
            
            
            column_well1time=self.cheese_df[f'well1time{col_num}'][0]
            if not np.isnan(column_well1time).any():
                start_idx=int((column_well1time-window)*self.pyFs)
                end_idx=int((column_well1time+window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                event_time_photometry_trace[f'pyData{col_num}'+'_beforewell1']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax2, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'well1time{col_num}'][0], before_window=window,
                                  after_window=window, fs=self.pyFs,color='green',Label=f'Trace{col_num+1} before well1')
            else:
                length=(window*2)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_beforewell1']=pd.Series(np.nan, index=range(length))
           
            column_leftwll1time=self.cheese_df[f'leftwell1time{col_num}'][0]
            if not np.isnan(column_leftwll1time).any():
                start_idx=int((column_leftwll1time-window)*self.pyFs)
                end_idx=int((column_leftwll1time+window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                event_time_photometry_trace[f'pyData{col_num}'+'_leftwell1']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax3, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'leftwell1time{col_num}'][0], before_window=window,
                                  after_window=window, fs=self.pyFs,color='green',Label=f'Trace{col_num+1} enter time')
            else:
                length=(window*2)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_leftwell1']=pd.Series(np.nan, index=range(length))   
           
            column_well2time=self.cheese_df[f'well2time{col_num}'][0]
            if not np.isnan(column_well2time).any():
                start_idx=int((column_well2time-window)*self.pyFs)
                end_idx=int((column_well2time+window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                event_time_photometry_trace[f'pyData{col_num}'+'_beforewell2']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax4, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'well2time{col_num}'][0], before_window=window,
                                  after_window=window, fs=self.pyFs,color='red',Label=f'Trace{col_num+1} before Well2')
            else:
                length=(window*2)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_beforewell2']=pd.Series(np.nan, index=range(length))
        plt.xlabel('Time (second)')
        plt.ylabel('Value')
        fig.suptitle('Event time traces '+self.animalID)
        plt.tight_layout()
        plt.show()
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+str(window)+'sec_twosides_window_event_traces.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        
        'filtered each event and plot average'
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_enter')]
        entertime_PETH = event_time_photometry_trace[filtered_columns]
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_beforewell1')]
        before_well1_PETH= event_time_photometry_trace[filtered_columns]
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_leftwell1')]
        left_well1_PETH= event_time_photometry_trace[filtered_columns]
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_beforewell2')]
        before_well2_PETH= event_time_photometry_trace[filtered_columns]
                                                                        
        fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2, figsize=(10, 10))
        fp.Plot_mean_With_CI_PSTH(entertime_PETH, window, window, self.animalID, meancolor='grey', stdcolor='lightgrey', ax=ax1)
        fp.Plot_mean_With_CI_PSTH(before_well1_PETH, window, window, self.animalID, meancolor='green', stdcolor='lightgreen', ax=ax2)
        fp.Plot_mean_With_CI_PSTH(left_well1_PETH, window, window, self.animalID, meancolor='green', stdcolor='lightgreen', ax=ax3)
        fp.Plot_mean_With_CI_PSTH(before_well2_PETH, window, window, self.animalID, meancolor='red', stdcolor='lightcoral', ax=ax4)
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+'_'+str(window)+'sec_window_twosides_event_average.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)

        '''save the pkl file for the PETH data with half window time specified'''
        filename=self.animalID+'_'+self.SessionID+'_'+str(window)+'sec_twosides_window_averaged_traces.pkl'
        self.event_time_photometry_trace=event_time_photometry_trace
        self.event_time_photometry_trace.to_pickle(os.path.join(self.result_path, filename))
        return self.event_time_photometry_trace
    
    def StartBox_twosides(self,before_window,after_window):
        event_time_photometry_trace = pd.DataFrame([])
        selected_columns = [col_name for col_name in self.photometry_df.columns if col_name.startswith('pyData') and col_name[6:].isdigit()]
        print ('selected_columns:',selected_columns)
        column_numbers = [int(col_name.replace('pyData', '')) for col_name in selected_columns]  
        '''This is to make a figure and plot all PETH traces on the same figure'''

        fig, ax  = plt.subplots(1, 1, figsize=(5, 5))
        for col_num in column_numbers:
            column_name = f'pyData{col_num}'
            column_photometry_data = self.photometry_df[column_name]
            
            column_entertime=self.cheese_df[f'entertime{col_num}'][0]
            print ('column_entertime--',f'entertime{col_num}--',column_entertime)
            if not np.isnan(column_entertime).any():
                start_idx=int((column_entertime-before_window)*self.pyFs)
                end_idx=int((column_entertime+after_window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                #print ('para_ENTER_photometry_data',para_event_photometry_data)
                event_time_photometry_trace[f'pyData{col_num}'+'_enter']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax, self.photometry_df[f'pyData{col_num}'],centre_time=
                                  self.cheese_df[f'entertime{col_num}'][0], before_window=before_window,
                                  after_window=after_window, fs=self.pyFs,color='grey',Label=f'Trace{col_num+1} enter time')
            else:
                length=(before_window+after_window)*self.pyFs
                event_time_photometry_trace[f'pyData{col_num}'+'_enter']=pd.Series(np.nan, index=range(length))
            
        plt.xlabel('Time (second)')
        plt.ylabel('Value')
        fig.suptitle('Event time traces '+self.animalID)
        plt.tight_layout()
        plt.show()
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+'_sec_twosides_window_SB_traces.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        
        'filtered each event and plot average'
        filtered_columns = [col for col in event_time_photometry_trace.columns if col.endswith('_enter')]
        entertime_PETH = event_time_photometry_trace[filtered_columns]
                                                                        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fp.Plot_mean_With_CI_PSTH(entertime_PETH, before_window, after_window, self.animalID, meancolor='grey', stdcolor='lightgrey', ax=ax)
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+'_'+'sec_window_twosides_SB_average.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)

        '''save the pkl file for the PETH data with half window time specified'''
        filename=self.animalID+'_'+self.SessionID+'_'+'sec_twosides_on_startbox.pkl'
        self.event_time_photometry_trace=event_time_photometry_trace
        self.event_time_photometry_trace.to_pickle(os.path.join(self.result_path, filename))
        return self.event_time_photometry_trace
    
    def plot_single_trial_2_rewards_PETH(self, trialIdx,before_well1_window, after_well2_window, color='blue', ax=None):
        'NEED TO MODIFY'
        py_signal = self.photometry_df[f'pyData{trialIdx}']
        well1time=self.cheese_df[f'well1time{trialIdx}'][0]
        well2time=self.cheese_df[f'well2time{trialIdx}'][0]
        if not np.isnan(well1time):
            start_idx=int((well1time-before_well1_window)*self.pyFs)
            event_time1 = 0
            print ('start_idx',start_idx)
            if not np.isnan(well2time):
                end_idx=int((well2time+after_well2_window)*self.pyFs)
                event_time2 =well2time-well1time
                print ('end_idx',end_idx)
            else:
                print ('Animal did not find the 2nd reward')
                end_idx=int((well1time+after_well2_window)*self.pyFs)
                event_time2 =0
                print ('end_idx',end_idx)
        else:
            print ('Animal found neither rewards')
            return -1
        
        py_signal_PETH=py_signal[start_idx:end_idx].reset_index(drop=True)
        num_samples = len(py_signal_PETH)
        time_in_seconds = np.linspace(-before_well1_window, event_time2+after_well2_window, num_samples)
        # If an 'ax' is provided, use it for plotting; otherwise, create a new figure and axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_in_seconds, py_signal_PETH, label='Optical Signal', color=color,linewidth=0.5)
        ax.axvline(x=event_time1, color='green', linestyle='--', label='Well1 Time')
        ax.axvline(x=event_time2, color='red', linestyle='--', label='Well2 Time')
        ax.set_xlabel('Time (second)')
        ax.set_ylabel('Value')
        ax.set_title('Optical trace when collecting 2 reward ' + self.animalID)
        # If 'ax' was provided, do not call plt.show() to allow the caller to display or save the figure as needed
        if ax is None:
            plt.show()
        return ax

    def find_peaks_in_SBtrials (self,plot=False):
        py_target_string='py'
        files = os.listdir(self.pySBFolder)
        filtered_files = [file for file in files if py_target_string in file]  
        min_distance = int(0.5 * self.pyFs)  # Minimum distance in samples
        for file in filtered_files:
            match = re.search(r'_(\d+)\.csv$', file)
            if match:
                target_index= match.group(1)
            # print('Target_index: ',target_index)
            # Read the CSV file with photometry read
            raw_signal,raw_reference=fp.read_photometry_data (self.pySBFolder, file, readCamSync=False,plot=False,sampling_rate=130)
            zdFF = fp.get_zdFF(raw_reference,raw_signal,smooth_win=10,remove=0,lambd=5e4,porder=1,itermax=50) 
            time = np.linspace(0, len(zdFF)/self.pyFs, len(zdFF))
            peaks, _ = signal.find_peaks(zdFF, distance=min_distance,height=1)  # Adjust height threshold as needed
            # Extract the peak values
            zdff_peak_values = zdFF[peaks]
            # Calculate the average of the peak values
            average_peak_value = np.mean(zdff_peak_values)
            num_peaks = len(peaks)
            peak_freq = num_peaks/(len(zdFF)/self.pyFs) #number of peak per seconds
            zdff_max = np.max(zdFF)
            # print("Average peak value:", average_peak_value)            
            # Construct the dictionary
            results_dict = {
                'zdff_peak_values': zdff_peak_values,
                'average_peak_value': average_peak_value,
                'num_peaks': num_peaks,
                'zdff_max': zdff_max,
                'peak_freq':peak_freq
            }
            # Save the dictionary to a pickle file
            pkl_filename = f'SB_peak_{self.animalID}_{self.SessionID}_trial{target_index}.pkl'
            pkl_filepath = os.path.join(self.result_path, pkl_filename)
            
            with open(pkl_filepath, 'wb') as pkl_file:
                pickle.dump(results_dict, pkl_file)
            if plot:    
                plt.figure(figsize=(10, 3))
                plt.plot(time, zdFF, label='Signal Trace')
                plt.plot(time[peaks], zdFF[peaks], 'rx', label='Peaks')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.title('Signal Trace with Peaks Labeled')
                plt.legend()
                plt.show()
        return -1
    