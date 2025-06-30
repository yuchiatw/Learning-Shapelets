import os
import sys 
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import ruptures as rpt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.preprocessing import normalize_data


mean_window_size = 4
seq_min = 15
padding_min = 3
step_min = 1

def preterm_pipeline(
    config={},
    meta_path = './data/filtered_clinical_data_v2.csv',
    strip_path='./data/filtered_strips_data_v2.json', 
    data_path = './data/preterm_v2.npz'
    ):
    '''
    Preterm birth data preprocessing pipeline.
    Input:
    - meta_path: Path to the metadata CSV file, including labeling information.
    - strip_path: Path to the JSON file containing the time series waveforms.
    '''
    if not config:
        config = {
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'norm_std': 'minmax',
            'norm_mode': 'local_before',
        }
    clinical = pd.read_csv(meta_path)
    clinical['Encounter_date'] = pd.to_datetime(clinical['Encounter_date'], errors='coerce')
    clinical['Delivery_date'] = pd.to_datetime(clinical['Delivery_date'], errors='coerce')
    clinical['EncData_ShiftedData'] = pd.to_datetime(clinical['EncDate_ShiftedDate'], errors='coerce')
    clinical['Pregnant_date'] = pd.to_datetime(clinical['Pregnant_date'], errors='coerce')
    
    label = clinical['Label'].values
    # Load the strip data.
    with open(strip_path, "r") as f:
        ua_list = json.load(f)
        
    # Apply trimming function to each strip and trimming to 20 minutes
    ua_list = [take_mean(strip) for strip in ua_list]
    ua_list = [trim_leading_zeros(strip) for strip in ua_list]
    
    for i in range(len(ua_list)):
        if len(ua_list[i]) < 20 * 60:
            ua_list[i] = np.pad(ua_list[i], [0, 20 * 60 - len(ua_list[i])], 'mean')
    ua_list = np.asarray(ua_list)
    ua_list = np.reshape(ua_list, (ua_list.shape[0], 1, ua_list.shape[1]))
    
    train_data, test_data, train_label, test_label  = \
        train_test_split(ua_list, label, test_size=config['test_ratio'] \
            if config.get('test_ratio') is not None else 0.2, shuffle=False)
    train_data, val_data, train_label, val_label = \
        train_test_split(train_data, train_label, test_size=config['val_ratio'] \
            if config.get('val_ratio') is not None else 0.2, shuffle=False)
    
    max_seq_len = train_data.shape[2]
    
    if config['norm_mode'] == 'global':
        mean, std = mean_std(train_data)
        mean = np.repeat(mean, max_seq_len).reshape(train_data.shape[1], max_seq_len)
        std = np.repeat(std, max_seq_len).reshape(train_data.shape[1], max_seq_len)
        train_data = mean_std_transform(train_data, mean, std)
        val_data = mean_std_transform(val_data, mean, std)
        test_data = mean_std_transform(test_data, mean, std)
    
    seq_length = int(60 * 4 / mean_window_size * (config['seq_min'] if config.get('seq_min') else seq_min))
    padding_threshold = int(60 * 4 / mean_window_size * (config['pad_min'] if config.get('pad_min') else padding_min))
    step = int(60 * 4 / mean_window_size * (config['step_min'] if config.get('step_min') else step_min))
    X_train, y_train = segmentation(train_data, train_label, 
        norm_std=config['norm_std'], norm_mode=config['norm_mode'],
            seq_length=seq_length, padding_threshold=padding_threshold, 
            step=step)
    X_val, y_val = segmentation(val_data, test_label, 
        norm_std=config['norm_std'], norm_mode=config['norm_mode'],
            seq_length=seq_length, padding_threshold=padding_threshold, 
            step=step)
    X_test, y_test = segmentation(test_data, test_label, 
        norm_std=config['norm_std'], norm_mode=config['norm_mode'],
            seq_length=seq_length, padding_threshold=padding_threshold,
            step=step)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    print(label_encoder.classes_)
    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)
    print(f'Train normal : abnormal - {np.sum(y_train == 0)} : {np.sum(y_train == 1)}')
    print(f'Val normal : abnormal - {np.sum(y_val == 0)} : {np.sum(y_val == 1)}')
    print(f'Test normal : abnormal - {np.sum(y_test == 0)} : {np.sum(y_test == 1)}')
    
    data = {}
    data['X_train'] = X_train
    data['X_val'] = X_val
    data['X_test'] = X_test
    data['y_train'] = y_train
    data['y_val'] = y_val
    data['y_test'] = y_test
    print(X_train.shape)
    # Save the dataset to disk
    np.savez(data_path, **data)
    return data
    
def detect_anomalous_start(signal, dynamic_threshold=0.1):
    """
    Detect if the beginning of a waveform has a different distribution than the rest.
    
    - Uses statistical tests and change-point detection.
    - The threshold determines what fraction of the signal is considered 'beginning'.
    """
    if len(signal) < 50:  # Skip too short waveforms
        return False
    
    # Dynamically determine the "beginning" range
    # start_range = max(50, int(len(signal) * dynamic_threshold))  # At least 50 points
    start_range = 4000 # At least 50 points
    
    start_segment = np.array(signal[:start_range])
    rest_segment = np.array(signal[start_range:])

    if len(rest_segment) == 0:  # If the waveform is too short after trimming
        return False
    
    # Statistical comparison
    mean_diff = abs(np.mean(start_segment) - np.mean(rest_segment))
    std_diff = abs(np.std(start_segment) - np.std(rest_segment))

    # Kolmogorov-Smirnov Test (check if distributions are different)
    ks_stat, ks_pval = stats.ks_2samp(start_segment, rest_segment)

    # Change-Point Detection (detect if there's a major shift)
    algo = rpt.Binseg(model="l2").fit(np.array(signal))
    change_points = algo.predict(n_bkps=1)  # Detect 1 breakpoint

    # Conditions for anomaly detection
    is_anomalous = (mean_diff > np.std(rest_segment) * 2) or \
                   (std_diff > np.std(rest_segment) * 2) 
                    # (change_points[0] < start_range * 1.5)  # Change-point occurs early
                   

    return is_anomalous

def take_mean(series, window_size = mean_window_size):
    n = len(series)
    reshaped = np.array(series)[:n - n % window_size].reshape(-1, window_size)
    mean_value = reshaped.mean(axis=1)
    return mean_value

def trim_leading_zeros(signal):
        """Remove continuous leading zeros until a nonzero value is encountered."""
        index = next((i for i, x in enumerate(signal) if x != 0), len(signal))

        return signal[index:index + 20 * 60] # keep 20 minutes of data
    

def segment_time_series(config, time_series, label, ID = None, segment_length = 200):
    n = len(time_series)
    # Number of complete segments
    num_segments = n // segment_length
    # Segment the series and assign the label to each segment
    segments = [
        time_series[i * segment_length:(i + 1) * segment_length]
        for i in range(num_segments)
    ]
    segment_labels = [label] * num_segments
    if ID:
        segment_ids = [ID] * num_segments
    return segments, segment_labels if ID is None else segments, segment_labels, segment_ids

def split_by_padding(time_series, padding_threshold):
    # Identify start and end indices of non-padding regions
    non_padding = np.where(time_series != 0, 1, 0)
    prepend = 1 if time_series[0]==0 else 0
    append = 1 if time_series[-1]==0 else 0
    changes = np.diff(non_padding, prepend=prepend, append=append)
    start_indices = np.where(changes == -1)[0]
    end_indices = np.where(changes == 1)[0]
    
    # Filter out zero-padding whichi is less than the threshold
    for start, end in zip(start_indices, end_indices):
        if end - start < padding_threshold and end < len(time_series):
            non_padding[start:end] = 1
    
    changes_region = np.diff(non_padding, prepend=0, append=0)
    start_indices = np.where(changes_region == 1)[0]
    end_indices = np.where(changes_region == -1)[0]
    
    regions = []
    for start, end in zip(start_indices, end_indices):
        regions.append(time_series[start:end])
    
    return regions

# Function to segment a single time series
def segment_time_series_excluding_padding(time_series, label, segment_length, 
                                          step = 60, padding_threshold = 180):
    # Split into non-padding regions
    non_padding_regions = split_by_padding(time_series, padding_threshold)
    
    segments = []
    segment_labels = []
    
    for region in non_padding_regions:
        # Segment each region
        region_segments = [
            region[i:i + segment_length]
            for i in range(0, len(region), step) if i + segment_length <= len(region)
        ]
        segments.extend(region_segments)
        segment_labels.extend([label] * len(region_segments))

    
    return segments, segment_labels 
def segmentation(
    tocometer, 
    labels, 
    seq_length = 900,
    step=60,
    padding_threshold = 180,
    norm_std='standard', 
    norm_mode='local_before',
):
    all_segments = []
    all_labels = []
    for ts, label in zip(tocometer, labels):
        if norm_mode=='local_before':
            ts, scaler = normalize_data(
                np.array(ts).reshape(1, -1), 
                mode=norm_std
            )
        ts = ts.reshape(-1)
        segments, segment_labels = \
            segment_time_series_excluding_padding(ts, label, 
                                                  segment_length=seq_length, 
                                                  step=step,
                                                  padding_threshold=padding_threshold
                                                )
        if norm_mode=='local_after':
            temp = []
            for i in range(len(segments)):
                norm_temp, scaler = normalize_data(
                    np.array(segments[i]).reshape(1, -1), 
                    mode=norm_std
                ) 
                temp.append(norm_temp)
                
            
            all_segments.extend(temp)
        else:
            all_segments.extend(segments)
            
        all_labels.extend(segment_labels)
        
    all_segments = np.array(all_segments).reshape(len(all_segments), 1, len(all_segments[0]))
    
    return all_segments, all_labels

def mean_std(train_data):
    m_len = np.mean(train_data, axis=2)
    mean = np.mean(m_len, axis=0)

    s_len = np.std(train_data)
    # std = np.max(s_len, axis=0)
    std = s_len
    
    return mean, std


def mean_std_transform(train_data, mean, std):
    '''
    Normalizing based on global mean and std.
    '''
    return (train_data - mean) / std

    
    
    
if __name__ == "__main__":
    config = {
        'test_ratio': 0.2,
        'val_ratio': 0.2,
        'norm_std': 'minmax',
        'norm_mode': 'local_before',
    }
    # preterm_pipeline(
    #     config=config, 
    #     meta_path='./data/filtered_clinical_data_v2.csv',
    #     strip_path='./data/filtered_strips_data_v2.json',
    #     data_path='./data/preterm_v2.npz'
    # )    
    preterm_pipeline(
        config=config, 
        meta_path='./data/filtered_clinical_data_v4.csv',
        strip_path='./data/filtered_strips_data_v4.json',
        data_path='./data/preterm_v4.npz'
    )    
    
    # preterm_pipeline(
    #     config=config, 
    #     meta_path='./data/filtered_clinical_data.csv',
    #     strip_path='./data/filtered_strips_data.json',
    #     data_path='./data/preterm.npz'
    # )    
    