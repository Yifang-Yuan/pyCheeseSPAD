# Photometry Recording Combined with Cheeseboard Task
 Processing SPAD and pyPhotometry data to get zscore and normalised traces.
## pyPhotometry data analysis
Analysis for pyPhotometry data is modified from:
https://github.com/katemartian/Photometry_data_processing

`photometry_functions.py` provides functions to read, batch-read pyPhotometry data that saved as .csv, it also includes codes to integrate photometry analysis to Cheeseboard task.

`PhotometryRunSingleTrace.py` is the main file to read and process pyPhotometry data, you'll be able to save normalised signal trace `"Green_traceAll.csv"`, reference trace `"Red_traceAll.csv"`, zscore trace `"Zscore_traceAll.csv"`. For photometry recording with behaviour tasks,a CamSync file if synchronisation is included: `"CamSync_photometry.csv"`.

**Sample data and example output**

For two samples in `pyPhotometrySampleData`, they are recorded an anaesthetised animal and during the wake-up period respectively. The animal was injected with GCamp8s(aav retro) into CA1. The signal has regular and high amplitude transients because a study found that we can detect 0.1Hz sharp waves by calcium recording in the hippocampus when the animal is anesthetised by isoflurane.

**Reference:** Anesthetics fragment hippocampal network activity, alter spine dynamics, and affect memory consolidation: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001146

![image](https://github.com/user-attachments/assets/aebf2bc2-d209-458f-a406-f83dae1e11b6)

## Cheeseboard task with photometry recording
A prerequisition is COLD pipeline (Daniel Lewis-Fallows,2024maybe) to process cheeseboard behavioural data. The output is .xlxs files that will be read in this part of analysis.
Example data structure of COLD output:

![image](https://github.com/user-attachments/assets/ff561104-9c71-4527-815f-6b0f532a63e5)

`pyCheeseSession.py` defines a Class of a session of cheeseboard task recording. 

`pyCheese_main.py` is the main function to define the pyCheeseSession Class and do analysis, can be used for developing and testing.

`pyCheeseBatch.py` will batch read multiple sessions of recording in a specific folder structure, and save PETH traces with a defined time window around reward collectoin, and data from start-box recording. 

![image](https://github.com/user-attachments/assets/7d5dcfec-3265-42ec-b183-ce8e21209c40)




#### Old testing files
**NOTE**: These two files are used for testing and simple analysis, for batch analysis, use the method above.

`pyCB_singleTrial_test.py` is only used to demonstrate synchronisation of behaviour and photometry data, and plot optical transient during collecting reward time for a single cheeseboard trial. It is not useful for averaging multiple trials or comparing across day sessions, etc. 

`pyCB_multiTrial_test.py` photometry data analysis for a day session with multiple training trials.

## SPAD-SPC imager data processing
More information about the SPC imager can be found in the README of this repository: 
https://github.com/MattNolanLab/SPAD_in_vivo

`mainAnalysisSPC.py` is the main function to run to process data recorded by the SPC imager. This file includes lines to process binary files, demodulate time division mode recordings. Saved results are: normalised signal trace `"Green_traceAll.csv"`, reference trace `"Red_traceAll.csv"`, zscore trace `"Zscore_traceAll.csv"`

`SPADreadBin.py` provides decoding functions for the binary data (,bin) saved by the SPC imager. Usually, we don't need to change anything here, functions are called by the `mianAnalysisSPC.py`.

`SPADdemod.py` provides demodulating functions to demodulate signal and unmix the neural signal trace and the reference trace.----For photometry imaging, we often have two channels, one is fluorescence signal that report neural activity, the other is a reference that does not change with neural activity but may report movement artefact. Therefore, time-division modulation or frequency modulation are used to modulate the two light channels. The modulation fuctions are not inbuild in the SPAD imaging system, we modulate two LEDs for excitation and two emissions are mixed in the raw imaging data. 



