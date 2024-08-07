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
    def __init__(self, pyFolder,CheeseFolder,COLD_filename,save_folder,animalID='mice1',SessionID='Day1',pySBFolder=None):
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
        filtered_files = [file for file in files if py_target_string in file]  
        cheeaseboard_session_data=fp.read_cheeseboard_from_COLD (self.CheeseFolder, self.COLD_resultfilename)
        photometry_df = pd.DataFrame([])
        cheese_df=pd.DataFrame([])
        for file in filtered_files:
            # Extract the last number from the file name
            match = re.search(r'_(\d+)\.csv$', file)
            if match:
                target_index= match.group(1)
            print('Target_index: ',target_index)
            # Read the CSV file with photometry read
            raw_signal,raw_reference,Cam_Sync=fp.read_photometry_data (self.pyFolder, file, readCamSync=True,plot=False,sampling_rate=130)
            zdFF = fp.get_zdFF(raw_reference,raw_signal,smooth_win=10,remove=0,lambd=5e4,porder=1,itermax=50) 
            filtered_files_sync = [file for file in files if sync_target_string in file]
            for Sync_file in filtered_files_sync:
                match_sync = re.search(r'sync_(\d+)\.csv$', Sync_file)
                if match_sync:
                    sync_index= match_sync.group(1)
                if sync_index == target_index:
                    print(Sync_file)
                    CamSync_LED=fp.read_Bonsai_Sync (self.pyFolder,Sync_file,plot=False)
                    zscore_sync,Sync_index_inCam,Sync_Start_time=fp.sync_photometry_Cam(zdFF,Cam_Sync,CamSync_LED,CamFs=self.CamFs)
                    zscore_series=pd.Series(zscore_sync)
                    zscore_series=zscore_series.reset_index(drop=True)
                    Sync_Start_time_series=pd.Series(Sync_Start_time)
                    entertime, well1time,well2time=fp.adjust_time_to_photometry(cheeaseboard_session_data,int(target_index),Sync_Start_time)
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
            self.photometry_df=photometry_df
            self.cheese_df=cheese_df
        return self.photometry_df,self.cheese_df
        
    def Plot_multiple_PETH_different_window(self,before_window,after_window):
        event_window_traces = pd.DataFrame([])
        selected_columns = [col_name for col_name in self.photometry_df.columns if col_name.startswith('pyData') and col_name[6:].isdigit()]
        print (selected_columns)
        column_numbers = [int(col_name.replace('pyData', '')) for col_name in selected_columns]  
        event_time = 0 
        '''This is to make a figure and plot all PETH traces on the same figure'''
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        for col_num in column_numbers:
            column_name = f'pyData{col_num}'
            column_photometry_data = self.photometry_df[column_name]
            column_well1time=self.cheese_df[f'well1time{col_num}'][0]
            if not np.isnan(column_well1time).any():
                start_idx=int((column_well1time-before_window)*self.pyFs)
                end_idx=int((column_well1time+after_window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                event_window_traces[f'pyData{col_num}'+'_1']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax, self.photometry_df[f'pyData{col_num}'],centre_time=
                                 self.cheese_df[f'well1time{col_num}'][0], before_window=before_window,
                                 after_window=after_window, fs=self.pyFs,color='green',Label=f'Trace{col_num+1} Well1time')
            column_well2time=self.cheese_df[f'well2time{col_num}'][0]
            if not np.isnan(column_well2time).any():
                start_idx=int((column_well2time-before_window)*self.pyFs)
                end_idx=int((column_well2time+after_window)*self.pyFs)
                para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
                event_window_traces[f'pyData{col_num}'+'_2']=para_event_photometry_data
                fp.PETH_plot_zscore_diff_window(ax, self.photometry_df[f'pyData{col_num}'],centre_time=
                                 self.cheese_df[f'well2time{col_num}'][0], before_window=before_window,
                                 after_window=after_window, fs=self.pyFs,color='red',Label=f'Trace{col_num+1} Well2time')
        ax.axvline(x=0, color='red', linestyle='--', label='Event Time') 
        plt.xlabel('Time (second)')
        plt.ylabel('Value')
        plt.title('Single Calcium traces while reaching well1 (green) and well2 (red) '+self.animalID)
        output_path = os.path.join(self.result_path, self.animalID+self.SessionID+str(before_window)+'sec_window_reward_single.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.legend()
        main_signal = event_window_traces.mean(axis=1)
        std_deviation = event_window_traces.std(axis=1)
         
        # Create the plot
        num_samples = len(main_signal)
        #time_in_seconds = np.arange(num_samples) / fs
        time_in_seconds = np.linspace(-before_window, after_window, num_samples)
        # Set the x-axis tick positions and labels
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(time_in_seconds,main_signal, label='Mean Signal', color='blue')
        ax.fill_between(time_in_seconds, main_signal - std_deviation, main_signal + std_deviation, color='lightblue', alpha=0.5, label='Standard Deviation')
        ax.axvline(x=event_time, color='red', linestyle='--', label='Event Time')
        ax.set_xlabel('Time (second)')
        ax.set_ylabel('Value')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.title('Mean Signal with Standard Deviation '+self.animalID)
        ax.legend()
        plt.show()
        output_path = os.path.join(self.result_path, self.animalID+'_'+self.SessionID+'_'+str(before_window)+'sec_window_reward_mean_std.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        
        '''save the pkl file for the PETH data with half window time specified'''
        filename=self.animalID+'_'+self.SessionID+'_'+str(before_window)+'sec_win_traces.pkl'
        self.event_window_traces=event_window_traces
        self.event_window_traces.to_pickle(os.path.join(self.result_path, filename))
        return self.event_window_traces
    
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

    def find_peaks_in_SBtrials (self):
        py_target_string='py'
        files = os.listdir(self.pySBFolder)
        filtered_files = [file for file in files if py_target_string in file]  
        min_distance = int(0.5 * self.pyFs)  # Minimum distance in samples
        for file in filtered_files:
            match = re.search(r'_(\d+)\.csv$', file)
            if match:
                target_index= match.group(1)
            print('Target_index: ',target_index)
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
            zdff_max = np.max(zdFF)
            print("Average peak value:", average_peak_value)            
            # Construct the dictionary
            results_dict = {
                'zdff_peak_values': zdff_peak_values,
                'average_peak_value': average_peak_value,
                'num_peaks': num_peaks,
                'zdff_max': zdff_max
            }
            # Save the dictionary to a pickle file
            pkl_filename = f'SB_peak_{self.animalID}_{self.SessionID}_trial{target_index}.pkl'
            pkl_filepath = os.path.join(self.result_path, pkl_filename)
            
            with open(pkl_filepath, 'wb') as pkl_file:
                pickle.dump(results_dict, pkl_file)
                
            plt.figure(figsize=(10, 3))
            plt.plot(time, zdFF, label='Signal Trace')
            plt.plot(time[peaks], zdFF[peaks], 'rx', label='Peaks')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('Signal Trace with Peaks Labeled')
            plt.legend()
            plt.show()
        return -1