# SPADPhotometryAnalysis
 Processing SPAD and pyPhotometry data to get zscore and normalised traces.
## pyPhotometry data analysis
Analysis for pyPhotometry data is modified from:
https://github.com/katemartian/Photometry_data_processing

`photometry_functions.py` provides functions to read, batch-read pyPhotometry data that saved as .csv, it also includes codes to integrate photometry analysis to Cheeseboard task.

`PhotometryTraceAnalysis.py` is the main file to read and process pyPhotometry data, you'll be able to save normalised signal trace, reference trace and zscore trace.

### Cheeseboard task with photometry recording
`pyCheese_singleTrial.py` and `pyCheese_multiTrial.py` provide photometry data analysis with cheeseboard task. 
A prerequisition is COLD pipeline (Lewis-Fallows 2024) to process cheeseboard behavioural data. 

## SPAD-SPC imager data processing
More information about the SPC imager can be found in the README of this repository: 
https://github.com/MattNolanLab/SPAD_in_vivo
