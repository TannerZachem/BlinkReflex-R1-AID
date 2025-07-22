# DataProcess.py
# This file loads the raw data from JSON, the annotations, and processes it for ML analysis
#Import necessary libraries
import json
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from glob import glob
import pickle
from matplotlib.patches import Rectangle

#Load Files
#JSON file containing the raw data
file = json.load(open(r'C:\Users\tjz5\Desktop\1-Projects\Blink-Protected\All_Blink.json'))
#Set up the DataFrame to hold the processed data
DF = pd.DataFrame(columns=["CaseID","Mode","Timestamp","TrialNumber","Sweep","DataLoc","Data","Gain","Stimuli"])
#Set annotations path to load the annotations
annotations_path = r'C:\Users\tjz5\Desktop\1-Projects\Blink-Protected\Annotations\all-annotations\*'
#Load file for surgical laterality
laterality_df = pd.read_csv(r'C:\Users\tjz5\Desktop\1-Projects\Blink-Protected\Code\Blink-laterality.csv')
# Map the laterality values to 'L' and 'R'
laterality_df['tumor_loc'] = laterality_df['tumor_loc'].map({1.0:"L",2.0:"R"})

#Function to get the end time of the stimuli
def get_stimuli_end_time(row, interpulse=0.0007):
    """ Calculate the end time of the last pulse train in a trial.
    Args:
        row (pd.Series): A row from the DataFrame containing trial data.
        interpulse (float): The time between pulses in seconds (default is 0.0007).
    Returns:
        float: The end time of the last pulse train in seconds.
    """
    a = row['Stimuli']   
    last_pulse_train_start = a['DiscreteStimuli'][-1]["Displacement"]
    last_pulse_start = a['DiscreteStimuli'][-1]["ElectricalPulses"][-1]["Displacement"]
    last_pulse_pulse_width = a['DiscreteStimuli'][-1]["ElectricalPulses"][-1]["PulseWidth"]
    last_pulse_end = last_pulse_start + last_pulse_pulse_width
    last_pulse_train_end = last_pulse_train_start + last_pulse_end + interpulse
    return last_pulse_train_end

#Convert Pixels to time from annotations (this data is not used in the ML model)
def pixels_to_time(x1,x2):
    """ Convert pixel coordinates to time in seconds.
    Args:
        x1 (float): The x-coordinate of the first pixel.
        x2 (float): The x-coordinate of the second pixel.
    Returns:
        tuple: A tuple containing the time in seconds for the first and second pixel.
    """
    px_t = 0.6024
    x0 = 13 
    t1 = (x1-x0)*px_t
    t2 = (x2-x0)*px_t
    return t1,t2

def get_annotation_data(annotation):
    """ Extracts relevant data from an annotation.
    Args:
        annotation (dict): A dictionary containing the annotation data.
    Returns:
        tuple: A tuple containing the annotation ID, start time, end time, label, and signal quality.
    """
    id = annotation['id']
    t1,t2,label,signal_quality = None,None,None,None
    for i in range(len(annotation['result'])):
        choice = annotation['result'][i]['type']
        if choice == 'choices':
            signal_quality = int(annotation['result'][i]['value']['choices'][0].split(':')[0])
            break
            
    if signal_quality and signal_quality !=1:
        for item in annotation['result']:
            if item['type'] == 'rectanglelabels':
                x1 = float(item['value']['x'])
                x2 = x1 + float(item['value']['width'])
                label = item['value']['rectanglelabels'][0]
                t1,t2 = pixels_to_time(x1,x2)
                break
    return id,t1,t2,label,signal_quality


def epoch_pad_resample(row,
                       window_s=0.035,   # 35 ms window
                       pad_s=0.005,      # mean of 5 ms prior
                       n_samples=448):   # fixed output length
    """ Process a single trial row to create a fixed-length epoch.
    Args:
        row (pd.Series): A row from the DataFrame containing trial data.
        window_s (float): Length of the epoch in seconds (default is 0.035).
        pad_s (float): Length of the padding in seconds (default is 0.005).
        n_samples (int): Number of samples in the output epoch (default is 448).
    Returns:
        np.ndarray: A resampled epoch of fixed length.
    """
    data      = np.asarray(row['Data'])
    tds      = row['TDS'] #Cadwell stores TDS: Trace Data Scalar
    data     = data * tds #Multiply by TDS to get data in Volts
    fs        = len(data) / row['Sweep']                # per-row Hz
    stim_idx  = int(round(row['stim_end_time'] * fs))
    
    #Epoch
    seg = data[stim_idx:]
    needed = window_s * fs - len(seg)
    if needed > 0:
        # how many pad samples
        pad_pts = int(round(needed))
        # compute mean of the last pad_s seconds before stim_idx
        src_pts  = int(round(pad_s * fs))
        src      = data[max(0, stim_idx-src_pts):stim_idx]
        pad_val  = src.mean() if src.size>0 else 0.0
        seg      = np.concatenate([seg, np.full(pad_pts, pad_val)])
    else:
        seg = seg[:int(round(window_s * fs))]
    
    #Resample to n_samples via linear interp
    t_old    = np.linspace(0, window_s, len(seg))
    t_new    = np.linspace(0, window_s, n_samples)
    seg_rs   = np.interp(t_new, t_old, seg)
    return seg_rs

def zscore_epoch(epoch):
    """ Z-score normalize an epoch.
    Args:
        epoch (list or np.ndarray): The epoch data to normalize.
    Returns:
        list: A list of z-score normalized values.
    """
    ep = np.asarray(epoch, dtype=float)
    mu, sigma = ep.mean(), ep.std()
    if sigma == 0:
        # if flat line, just subtract mean (all zeros)
        return (ep - mu).tolist()
    return ((ep - mu) / sigma).tolist()

#Initialize a dictionary to hold patient IDs
p_dict = {}

#Iterate through the cases in the JSON file and extract relevant data
cases = file['Cases']
for c in cases:
    caseID = c["ID"]
    modes = c["Modes"]
    p_dict[caseID] = c["PatientID"]
    for mode in modes:
        mName = mode["Name"]
        trials = mode["Trials"]
        for trial in trials:
            timestamp = trial["Timestamp"]
            trialNumber = trial["TrialNumber"]
            stim = trial["Stimuli"]
            traces = trial["Traces"]
            baseline = trial["Baseline"]
            for trace in traces:
                channel = trace["Channel"]
                dataLoc = channel["Name"]
                gain = channel["Gain"]
                data = trace["TraceData"]
                tds = trace["TraceDataScalar"]
                sweep = trace["Sweep"]
                newRow = pd.DataFrame([[caseID,mName,timestamp,trialNumber,sweep,dataLoc,data,tds,gain,stim,baseline]],columns=["CaseID","Mode","Timestamp","TrialNumber","Sweep","DataLoc","Data","TDS","Gain","Stimuli","Baseline"])
                DF = pd.concat([DF,newRow],ignore_index=True)
 
data = DF['Data']
data = data.apply(lambda x: np.array(x))
DF['Mode'] = DF['Mode'].apply(lambda x: x[0])

#List to hold annotations
annotations = []

for file in glob(annotations_path):
    a = json.load(open(file))
    annotations.append(a)

#Get CaseIDs
caseIds = DF["CaseID"].unique()

DF['MRN'] = DF['CaseID'].map(p_dict)
DF = DF.merge(laterality_df[['MRN','tumor_loc']], on='MRN', how='left')

# Initialize lists to hold processed data
caseID_list = []
annotationID_list = []
mode_list = []
timestamp_list = []
data_list = []
signal_quality_list = []
label_list = []
t1_list = []
t2_list = []
num_not_found = 0 
for annotation in annotations: 
    try:
        aID,t1,t2,label,signal_quality = get_annotation_data(annotation)
        s = annotation['task']['data']['image']
        g = s.split('/')[-1]
        f = g.split('.')[0]
        d = f.split('-')
        #handle the different saving on some files
        if len(d) != 7:
            print("caught")
            d1 = d[-1].split("_")
            d = d[:-1] + d1
        cID = '-'.join(d[0:5])
        modeI = d[5]
        tsI = int(d[6])

        DF_slice = DF[(DF['CaseID'] == cID) & (DF['Timestamp'] == tsI)]
    except Exception as e:
        print(e)
    if DF_slice.empty:
        print('Data not found for annotation: {:}'.format(aID))
        num_not_found += 1
        continue
    else:
        data = np.array(DF_slice['Data'].values[0])
        caseID_list.append(cID)
        annotationID_list.append(aID)
        mode_list.append(modeI)
        timestamp_list.append(tsI)
        data_list.append(data)
        signal_quality_list.append(signal_quality)
        label_list.append(label)
        t1_list.append(t1)
        t2_list.append(t2)

#Ensure all data is found
print(num_not_found)

annotation_df = pd.DataFrame({
    'CaseID': caseID_list,
    "Timestamp": timestamp_list,
    'signal_quality': signal_quality_list,
    'Mode': mode_list,
    'label': label_list,
    't1': t1_list,
    't2': t2_list
})

blinkDF = DF.merge(annotation_df, on=['CaseID',"Mode","Timestamp"], how="left")
#Obtain the ipsilateral data
DF_ipsi = blinkDF[blinkDF["Mode"]==blinkDF["tumor_loc"]]
DF_ipsi['DataLoc'] = DF_ipsi['DataLoc'].str.strip().str.upper()
#Some data has different naming conventions, so we map them to a standard format
mapping = {
    "L OCULI":"L OCULI",
    "R OCULI":"R OCULI",
    "R OCULI - R OCULI'":"R OCULI",
    "RIGHT OCULI":"R OCULI"
}

DF_ipsi["DataLoc"] = DF_ipsi['DataLoc'].map(mapping)
DF_ipsi = DF_ipsi.dropna(subset=['DataLoc']).reset_index(drop=True)

#Get laterality of recording
DF_ipsi['DataSide'] = DF_ipsi["DataLoc"].str[0]
# Filter to keep only the ipsilateral data, Mode is the stimulation side
DF_signal = DF_ipsi[DF_ipsi["Mode"]==DF_ipsi["DataSide"]]

DF_signal['stim_end_time'] = DF_signal.apply(get_stimuli_end_time, axis=1)

#Some exploratory analysis to determine the best epoch length
#Get the lengths of the signal data
lengths = DF_signal['Data'].apply(len)
sweeps = DF_signal["Sweep"]
print("Sweeps:",sweeps.value_counts())
#See the unique lengths
unique_lengths = sorted(lengths.unique())
print("Unique lengths:", unique_lengths)
#Counts
counts = lengths.value_counts().sort_index()
print("\nCounts per length:\n", counts)
#Get time after stim for all rows
DF_signal['time_after'] = DF_signal['Sweep'] - DF_signal['stim_end_time']
#See summary stats
print("Overall time_after stats:")
print(DF_signal['time_after'].describe(), '\n')
#Percentiles 
pcts = DF_signal['time_after'].quantile([0.01,0.05,0.10,0.25,0.50])
print("Key percentiles of time_after (s):\n", pcts, '\n')
#Correlation between Sweep and stim time to see if there are any interesting relationships
corr = DF_signal['Sweep'].corr(DF_signal['stim_end_time'])
print(f"Pearson r (Sweep vs. stim_end): {corr:.3f}\n")
#Scatterplot of stim_end vs. Sweep
plt.figure(figsize=(6,6))
plt.scatter(DF_signal['stim_end_time'], DF_signal['Sweep'], alpha=0.3)
plt.plot([0, DF_signal['Sweep'].max()],[0, DF_signal['Sweep'].max()],'k--', linewidth=1)
plt.xlabel('Stim end time (s)')
plt.ylabel('Sweep duration (s)')
plt.title('Stim-end vs. Sweep')
plt.show()
#Histogram of time_after
plt.figure(figsize=(6,4))
plt.hist(DF_signal['time_after'], bins=50, edgecolor='k')
plt.xlabel('Time after stim (s)')
plt.ylabel('Count')
plt.title('Distribution of available post-stim time')
plt.show()

#Create a deep copy of the DataFrame for processing epochs
DF = DF_signal.copy()
#Epoch and resample the data
DF['epoch_rs'] = DF.apply(lambda r: epoch_pad_resample(r), axis=1)

#Verify lengths of epochs
lengths = DF['epoch_rs'].apply(len).unique()
print("Fixed epoch length:", lengths)  # [448]


#Normalize the epochs using z-score
DF['epoch_norm'] = DF['epoch_rs'].apply(zscore_epoch)

# Create a DataFrame with the processed data, with only relevant columns
dataDF = pd.DataFrame({
    'DataNorm'      : DF['epoch_norm'],
    'CaseID'        : DF['CaseID'],
    'SignalQuality' : DF['signal_quality']})
#Reset the index of the DataFrame
dataDF = dataDF.reset_index()
#Save the processed DataFrame to a pickle file
with open(r"C:\Users\tjz5\Desktop\1-Projects\Blink-Protected\Code\DF-Normalized-Epochs.pkl", "wb") as f:
    pickle.dump(dataDF, f)
#This pickle can be laoded in the file "R1AID-ML.py" to train and test the model. 

#--------------------------------------------------------------------------------------------------------
#Create Figure 2 of the Manuscript
#This function shows the data processing steps for a single epoch

def plot_epoch_simple(row,
                      window_s=0.035,    # 35 ms window
                      pad_s=0.005,       # 5 ms prior
                      n_samples=448):    # fixed output length
    """ Plot the processing steps for a single epoch.
    Args:
        row (pd.Series): A row from the DataFrame containing trial data.
        window_s (float): Length of the epoch in seconds (default is 0.035).
        pad_s (float): Length of the padding in seconds (default is 0.005).
        n_samples (int): Number of samples in the output epoch (default is 448).
    """
    # Prepare data 
    data     = np.asarray(row['Data'])
    tds     = row['TDS']
    data    = data * tds  # normalize by TDS
    total_s  = row['Sweep']
    fs       = len(data) / total_s
    stim_idx = int(round(row['stim_end_time'] * fs))
    pts_win  = int(round(window_s * fs))

    #Epoch
    seg = data[stim_idx : stim_idx + pts_win]
    if len(seg) < pts_win:
        pad_pts = pts_win - len(seg)
        src_pts = int(round(pad_s * fs))
        src     = data[max(0,stim_idx-src_pts) : stim_idx]
        val     = src.mean() if src.size>0 else 0.0
        seg     = np.concatenate([seg, np.full(pad_pts, val)])

    #Resample
    t_old  = np.linspace(0, window_s, len(seg))       # sec
    t_new  = np.linspace(0, window_s, n_samples)      # sec
    seg_rs = np.interp(t_new, t_old, seg)

    #Z-Score
    seg_z  = (seg_rs - seg_rs.mean()) / seg_rs.std(ddof=0)

    #Convert times to ms
    t_full_ms = np.linspace(0, total_s*1000, len(data))
    t_epoch_ms = t_new * 1000

    #Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=False)

    # Panel 1: raw + window box
    axs[0].plot(t_full_ms, data, linewidth=0.8)
    y0, y1 = axs[0].get_ylim()
    start_ms = stim_idx/fs * 1000
    box = Rectangle((start_ms, y0),
                    window_s*1000, y1-y0,
                    facecolor='darkgray', alpha=0.3)
    axs[0].add_patch(box)
    axs[0].set(title="1) Raw Signal with Epoch Window",
               xlabel="Time (ms)", ylabel="Amplitude")

    # Panel 2: final resampled epoch
    axs[1].plot(t_epoch_ms, seg_rs, linewidth=1.0)
    axs[1].set(title="2) Resampled Epoch",
               xlabel="Time (ms)", ylabel="Amplitude")

    # Panel 3: z-score normalized
    axs[2].plot(t_epoch_ms, seg_z, linewidth=1.0)
    axs[2].set(title="3) Z-score Normalized",
               xlabel="Time (ms)", ylabel="Z-score")

    plt.tight_layout()
    plt.savefig("epoch_plot.png", dpi=1200)
    plt.savefig('epoch-plot.svg', format='svg')
    plt.show()

# get one example row (as a Series)
n = 99 #manually set to get a different example (Obvious R1)
example = DF_signal[DF_signal['signal_quality']==2.0].sample(1, random_state=n).iloc[0]
plot_epoch_simple(example)
