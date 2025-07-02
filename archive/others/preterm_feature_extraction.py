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
    
    data = np.load('./data/ECG5000.npz')
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
    window_step = 1
    
    x_train_split = sliding_window_view(x_train, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    x_val_split = sliding_window_view(x_val, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    x_test_split = sliding_window_view(x_test, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
    print(x_train_split.shape)
    # print(x_train_split.shape)
    # print(x_val_split.shape)
    # print(x_test_split.shape)
    num_windows = x_test_split.shape[1]
    x_train_split = x_train_split.reshape(num_train * num_windows, window_size, in_channels)
    x_val_split = x_val_split.reshape(num_val * num_windows, window_size, in_channels)
    x_test_split = x_test_split.reshape(num_test * num_windows, window_size, in_channels)
    print(x_train_split.shape)
    plt.figure(figsize=(15, 10))
    random_indices = np.random.choice(x_train_split.shape[0], 5, replace=False)
    # Remove rows where all values are the same (constant signal)
    # Define a variance threshold (adjust based on your data)
    # variance_threshold = 0.005  # Change this value as needed
    
    cfg_file = tsfel.get_features_by_domain()
    X_train_split = tsfel.time_series_features_extractor(cfg_file, x_train_split)
