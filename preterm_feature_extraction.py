import tsfel
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import sys
import os
import matplotlib.pyplot as plt
import re

def feature_extraction_selection(X_train, threshold=0):
    corr_features, X_train = tsfel.correlated_features(X_train, drop_correlated=True)
    selector = VarianceThreshold(threshold=threshold)
    X_train = selector.fit_transform(X_train)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, corr_features, selector, scaler

def extraction_pipeline(Feature, corr_features, selector, scaler):
    Feature.drop(corr_features, axis=1, inplace=True)
    Feature = selector.transform(Feature)
    nFeature = scaler.transform(Feature.values)
    return nFeature
def clean_feature_name(feature):
    """
    Removes numeric prefixes and numeric suffixes from a TSFEL feature name.
    Example: '0_ECDF_0' -> 'ECDF'
    """
    # Step 1: Remove numeric prefix (e.g., "0_")
    feature = re.sub(r'^\d+_', '', feature)  
    
    # Step 2: Remove numeric suffix (e.g., "_0", "_1", etc.)
    feature = re.sub(r'_\d+$', '', feature)
    
    return feature

if __name__ == '__main__':
    
    data = np.load('./data/preterm_v3.npz')
    x_train = data['X_train'].transpose(0, 2, 1)
    x_val = data['X_val'].transpose(0, 2, 1)
    x_test = data['X_test'].transpose(0, 2, 1)
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    # Plot several sequences in x_train and save the image
    plt.figure(figsize=(15, 10))
    for i in range(10):  # Plot first 5 sequences
        plt.subplot(10, 1, i + 1)
        plt.plot(x_train[i, :, 0])
        plt.title(f'Sequence {i + 1}')
    plt.tight_layout()
    plt.savefig('x_train_sequences.png')
    plt.close()
    num_classes = len(set(y_train))
    num_train, len_ts, in_channels = x_train.shape
    num_val = x_val.shape[0]
    num_test = x_test.shape[0]
    
    # X_train = tsfel.time_series_features_extractor(cfg_file, x_train)
    window_size = 50
    window_step = 10
    
    x_train_split = sliding_window_view(x_train, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    x_val_split = sliding_window_view(x_val, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    x_test_split = sliding_window_view(x_test, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    
    # print(x_train_split.shape)
    # print(x_val_split.shape)
    # print(x_test_split.shape)
    num_windows = x_test_split.shape[1]
    x_train_split = x_train_split.reshape(num_train * num_windows, window_size, in_channels)
    x_val_split = x_val_split.reshape(num_val * num_windows, window_size, in_channels)
    x_test_split = x_test_split.reshape(num_test * num_windows, window_size, in_channels)
    plt.figure(figsize=(15, 10))
    random_indices = np.random.choice(x_train_split.shape[0], 5, replace=False)
    # Remove rows where all values are the same (constant signal)
    # Define a variance threshold (adjust based on your data)
    # variance_threshold = 0.005  # Change this value as needed
    
    cfg_file = tsfel.get_features_by_domain()
    # # Mask: Keep rows where variance is above the threshold
    # non_low_variance_mask = np.var(x_train_split.reshape(num_train * num_windows, window_size), axis=1) > variance_threshold

    # # Filter the dataset
    # x_train_filtered = x_train_split[non_low_variance_mask]
    
    # # non_constant_mask = np.var(x_train_split.reshape(num_train * num_windows, window_size), axis=1) > 0

    # # Keep only non-flat signals
    # X_train_filtered = tsfel.time_series_features_extractor(cfg_file, x_train_filtered)
    # corr_features, X_train = tsfel.correlated_features(X_train_filtered, drop_correlated=True)
    # print(corr_features)
    
   # Apply cleaning to all selected features
    # clean_selected_features = [clean_feature_name(feat) for feat in corr_features]
    # print(clean_selected_features)
    clean_selected_features = ['Average power', 'ECDF Percentile Count', 
                               'LPCC', 'LPCC', 'LPCC', 'LPCC', 'LPCC', 
                               'Median', 'Root mean square', 'Spectral distance', 
                               'Spectral roll-off', 'Spectral skewness', 'Spectral slope', \
                                'Spectral spread', 'Standard deviation', 'Sum absolute diff', 
                                'Wavelet absolute mean_25.0Hz', 'Wavelet absolute mean_3.12Hz', 
                                'Wavelet absolute mean_3.57Hz', 'Wavelet absolute mean_4.17Hz', 
                                'Wavelet absolute mean_5.0Hz', 'Wavelet absolute mean_6.25Hz', 
                                'Wavelet absolute mean_8.33Hz', 'Wavelet energy_25.0Hz', 'Wavelet energy_3.12Hz', 
                                'Wavelet energy_3.57Hz', 'Wavelet energy_4.17Hz', 'Wavelet energy_5.0Hz', 
                                'Wavelet energy_6.25Hz', 'Wavelet energy_8.33Hz', 'Wavelet standard deviation_12.5Hz', 
                                'Wavelet standard deviation_2.78Hz', 'Wavelet standard deviation_25.0Hz', 'Wavelet standard deviation_3.12Hz', 
                                'Wavelet standard deviation_3.57Hz', 'Wavelet standard deviation_4.17Hz', 'Wavelet standard deviation_5.0Hz', 
                                'Wavelet standard deviation_6.25Hz', 'Wavelet standard deviation_8.33Hz', 'Wavelet variance_2.78Hz', 
                                'Wavelet variance_3.12Hz', 'Wavelet variance_3.57Hz', 'Wavelet variance_4.17Hz', 'Wavelet variance_5.0Hz', 
                                'Wavelet variance_6.25Hz', 'Wavelet variance_8.33Hz']
    # clean_selected_features = ['Average power', 'ECDF Percentile Count', 'Median', 'Standard deviation']
    # print(clean_selected_features)
    # print(X_train.columns)
    # Disable all features in cfg_file first
    for domain in cfg_file.keys():
        for feature in cfg_file[domain]:
            cfg_file[domain][feature]["use"] = "no"  # Ensure correct format

    # Enable only the selected features
    for domain in cfg_file.keys():
        for feature in cfg_file[domain]:
            if feature in clean_selected_features:
                cfg_file[domain][feature]["use"] =  "yes"
    # print(cfg_file)
    X_train_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_train_split)
    print(X_train_split_filtered.shape)
    # X_train_final, corr_features, selector, scaler = feature_extraction_selection(X_train_filtered)
    # print(corr_features)
    # print(X_train_final)
    # X_train_split = tsfel.time_series_features_extractor(cfg_file, x_train_split)
    # X_val_split = tsfel.time_series_features_extractor(cfg_file, x_val_split)
    # X_test_split = tsfel.time_series_features_extractor(cfg_file, x_test_split)
    # X_train_split_filtered, corr_features, selector, scaler = feature_extraction_selection(X_train_split)
    # X_val_split_filtered = extraction_pipeline(X_val_split, corr_features, selector, scaler)
    # X_test_split_filtered = extraction_pipeline(X_test_split, corr_features, selector, scaler)
    
    
    # X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
    # X_val_split_filtered = X_val_split_filtered.reshape(num_val, num_windows, -1)
    # X_test_split_filtered = X_test_split_filtered.reshape(num_test, num_windows, -1)
    # num_features = X_train_split_filtered.shape[-1]
    # if np.isnan(X_train_split_filtered).any():
    #     print("There are NaN values in X_train_split_filtered")
    # else:
    #     print("No NaN values in X_train_split_filtered")
    # # print(X_train_split_filtered)